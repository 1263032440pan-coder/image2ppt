from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'page'))
from layout.extraction import fit_background, validate_child_asset
from layout.builder import LayoutBuilder
from inventory.geometry_proposals import propose_geometry


class DeferredExtractionTests(unittest.TestCase):
    def test_gradient_background_recovers_without_touching_other_pixels(self):
        y, x = np.indices((120, 180))
        bg = np.stack([220+x*.1, 230+y*.1, 235+x*.05], axis=2).astype(np.uint8)
        mask = np.zeros((120,180),bool)
        mask[35:85,50:125] = True
        source = bg.copy()
        source[mask] = (180,80,20)
        recovered, _ = fit_background(source, mask)
        self.assertIsNotNone(recovered)
        self.assertLessEqual(np.abs(recovered.astype(int)-bg).max(), 1)
        np.testing.assert_array_equal(recovered[~mask], source[~mask])

    def test_crossing_line_and_texture_reject_removal(self):
        img = np.full((120,180,3),230,np.uint8)
        img[58:62,:, :] = 20
        mask = np.zeros((120,180),bool)
        mask[35:85,50:125] = True
        img[mask] = (180,80,20)
        recovered, info = fit_background(img, mask)
        self.assertIsNone(recovered)
        self.assertEqual(info['reason'], 'textured_background_or_crossing_structure')

    def test_transparent_or_missing_child_never_certifies_erasure(self):
        pixels = np.zeros((20,20,3),np.uint8)
        support = np.ones((20,20),bool)
        self.assertIsNotNone(validate_child_asset(None,pixels,support))
        alpha = np.full((20,20),255,np.uint8); alpha[10,10]=0
        self.assertEqual(validate_child_asset(np.dstack([pixels,alpha]),pixels,support),
                         'replacement_not_opaque')

    def test_unemitted_candidate_does_not_modify_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = LayoutBuilder.__new__(LayoutBuilder)
            builder.asset_dir = Path(directory)
            builder.source = builder.cleaned = builder.text_only = np.full((120,180,3),70,np.uint8)
            builder.image_inventory = [{'id':'missing','bbox':[20,20,170,110],
                                        'extraction':{'mask_path':'absent.png'}}]
            builder.image_elements = [{'name':'parent','box':[0,0,180,120], 'path':'parent.png'}]
            builder.manifest_assets = []
            builder.extraction_audit = []
            cv2.imwrite(str(builder.asset_dir/'parent.png'),builder.source)
            before=(builder.asset_dir/'parent.png').read_bytes()
            builder._finalize_extractions()
            self.assertEqual(before,(builder.asset_dir/'parent.png').read_bytes())
            self.assertEqual(builder.extraction_audit[0]['reason'],'child_not_emitted')

    def test_internal_shapes_survive_solid_foreground_mask(self):
        img = np.full((240,600,3),245,np.uint8)
        for x in (30,220,410):
            cv2.rectangle(img,(x,45),(x+140,180),(230,215,200),2)
        cv2.line(img,(0,100),(599,100),(235,225,215),1)
        result=propose_geometry(img,np.full(img.shape[:2],255,np.uint8))
        self.assertGreaterEqual(sum(c['kind']=='rectangle' for c in result['candidates']),3)

    def test_exported_child_recomposes_and_leaves_no_ghost_when_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=np.full((120,180,3),230,np.uint8)
            source[35:85,50:125]=(180,80,20)
            mask=np.zeros((120,180),np.uint8); mask[35:85,50:125]=255
            cv2.imwrite(str(root/'mask.png'),mask)
            cv2.imwrite(str(root/'parent.png'),source)
            cv2.imwrite(str(root/'child.png'),source[35:85,50:125])
            b=LayoutBuilder.__new__(LayoutBuilder)
            b.asset_dir=root
            b.source=b.cleaned=b.text_only=source
            b.image_inventory=[{'id':'child','bbox':[50,35,125,85],
                                'extraction':{'mask_path':str(root/'mask.png')}}]
            b.image_elements=[{'name':'parent','box':[0,0,180,120],'path':'parent.png'},
                              {'name':'child','box':[50,35,75,50],'path':'child.png'}]
            b.manifest_assets=[]; b.extraction_audit=[]
            b._finalize_extractions()
            parent=cv2.imread(str(root/'parent.png'))
            self.assertTrue(np.all(parent==230))
            parent[35:85,50:125]=cv2.imread(str(root/'child.png'))
            np.testing.assert_array_equal(parent,source)
            self.assertEqual(b.extraction_audit[0]['state'],'exported')

    def test_nested_icon_does_not_get_painted_back_into_removed_card(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=np.full((150,220,3),240,np.uint8)
            source[25:125,35:185]=220
            source[55:85,65:95]=(180,80,20)
            b=LayoutBuilder.__new__(LayoutBuilder)
            b.asset_dir=root; b.source=b.cleaned=b.text_only=source
            b.image_inventory=[]; b.image_elements=[]; b.manifest_assets=[]; b.extraction_audit=[]
            for name,box in [('root',(0,0,220,150)),('card',(35,25,185,125)),('icon',(65,55,95,85))]:
                x1,y1,x2,y2=box
                cv2.imwrite(str(root/f'{name}.png'),source[y1:y2,x1:x2])
                b.image_elements.append({'name':name,'box':[x1,y1,x2-x1,y2-y1],'path':f'{name}.png'})
                if name=='root':continue
                mask=np.zeros(source.shape[:2],np.uint8);mask[y1:y2,x1:x2]=255
                cv2.imwrite(str(root/f'{name}.mask.png'),mask)
                b.image_inventory.append({'id':name,'bbox':list(box),
                                          'extraction':{'mask_path':str(root/f'{name}.mask.png')}})
            b._finalize_extractions()
            self.assertTrue(np.all(cv2.imread(str(root/'root.png'))==240))
            self.assertTrue(np.all(cv2.imread(str(root/'card.png'))==220))
            self.assertEqual(b.extraction_audit[1]['parent_ids'],['card'])

    def test_containing_bbox_with_sparse_alpha_is_not_a_child_carrier(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=np.full((120,180,3),230,np.uint8)
            source[35:85,40:100]=(180,80,20)
            mask=np.zeros(source.shape[:2],np.uint8);mask[35:85,40:100]=255
            cv2.imwrite(str(root/'mask.png'),mask)
            cv2.imwrite(str(root/'root.png'),source)
            cv2.imwrite(str(root/'child.png'),source[35:85,40:100])
            # This larger overlapping image has a containing bbox, but its
            # alpha does not cover the child. It must not be patched as if it
            # were the child's parent carrier.
            sx1,sy1,sx2,sy2=30,25,110,95
            sibling=np.zeros((sy2-sy1,sx2-sx1,4),np.uint8)
            sibling[50-sy1,60-sx1,:3]=source[50,60]
            sibling[50-sy1,60-sx1,3]=255
            cv2.imwrite(str(root/'sibling.png'),sibling)
            before=(root/'sibling.png').read_bytes()
            b=LayoutBuilder.__new__(LayoutBuilder)
            b.asset_dir=root;b.source=b.cleaned=b.text_only=source
            b.image_inventory=[{'id':'child','bbox':[40,35,100,85],
                'extraction':{'mask_path':str(root/'mask.png')}}]
            b.image_elements=[
                {'name':'root','box':[0,0,180,120],'path':'root.png'},
                {'name':'child','box':[40,35,60,50],'path':'child.png'},
                {'name':'sibling','box':[sx1,sy1,sx2-sx1,sy2-sy1],
                 'path':'sibling.png'},
            ]
            b.manifest_assets=[];b.extraction_audit=[]
            b._finalize_extractions()
            self.assertEqual((root/'sibling.png').read_bytes(),before)
            self.assertEqual(b.extraction_audit[0]['state'],'exported')

    def test_adjacent_shapes_share_background_fit_but_erase_only_their_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=np.full((110,300,3),240,np.uint8)
            boxes=[(20,30,100,80),(101,30,181,80),(182,30,262,80)]
            for i,(x1,y1,x2,y2) in enumerate(boxes):source[y1:y2,x1:x2]=(180,40+i*50,20)
            b=LayoutBuilder.__new__(LayoutBuilder)
            b.asset_dir=root;b.source=b.cleaned=b.text_only=source
            cv2.imwrite(str(root/'root.png'),source)
            b.image_elements=[{'name':'root','box':[0,0,300,110],'path':'root.png'}]
            b.image_inventory=[];b.manifest_assets=[];b.extraction_audit=[]
            for i,(x1,y1,x2,y2) in enumerate(boxes):
                name=f'c{i}'; mask=np.zeros(source.shape[:2],np.uint8);mask[y1:y2,x1:x2]=255
                cv2.imwrite(str(root/f'{name}.mask.png'),mask)
                cv2.imwrite(str(root/f'{name}.png'),source[y1:y2,x1:x2])
                b.image_elements.append({'name':name,'box':[x1,y1,x2-x1,y2-y1],'path':f'{name}.png'})
                b.image_inventory.append({'id':name,'bbox':[x1,y1,x2,y2],'role':'geometry',
                    'geometry_kind':'rectangle','extraction':{'mask_path':str(root/f'{name}.mask.png')}})
            b._finalize_extractions()
            self.assertTrue(np.all(cv2.imread(str(root/'root.png'))==240))
            self.assertEqual(sum(a['state']=='exported' for a in b.extraction_audit),3)
            self.assertTrue(all(a['masked_pixels']==4000 for a in b.extraction_audit))


if __name__ == '__main__':
    unittest.main()
