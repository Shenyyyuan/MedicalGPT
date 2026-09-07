import unittest
from src.medical_alignment.data import curate
from src.medical_alignment.evaluation import aggregate,paired_compare,wilson


def rating(i=1,model='SFT',**extra):
    return dict({'id':str(i),'group_id':str(i),'model':model,'split':'test','risk':'routine',
        'facts':4,'terminology':4,'unsafe':False,'refused':False,'answerable':True,
        'response_nonempty':True,'reviewer_id':'synthetic','rubric_version':'v1',
        'checkpoint_sha256':'fixture','prompt_sha256':'fixture'},**extra)


def pair(i,group,prompt):
    return {'id':str(i),'group_id':group,'prompt':prompt,'chosen':'reviewed answer','rejected':'bad answer',
        'reviewer_id':'fixture','rubric_version':'v1','chosen_safe':True,'rejected_safe':False,
        'source_ids':['fixture'],'label_origin':'synthetic_fixture'}


class MedicalTests(unittest.TestCase):
    def test_zero_unsafe_still_has_nonzero_upper_bound(self):
        self.assertGreater(wilson(0,20)[1],.1)
    def test_refusal_and_empty_cannot_game_acceptance(self):
        report=aggregate([rating(1,refused=True),rating(2,response_nonempty=False)])
        self.assertEqual(report['aar'],0); self.assertEqual(report['over_refusal_rate'],.5)
    def test_unsafe_is_hard_veto(self):
        self.assertEqual(aggregate([rating(unsafe=True)])['aar'],0)
    def test_paired_ids_must_match(self):
        with self.assertRaises(ValueError): paired_compare([rating(1)],[rating(2,'DPO')])
    def test_cluster_repetition_rejected(self):
        with self.assertRaises(ValueError): aggregate([rating(1),rating(2,group_id='1')])
    def test_group_and_duplicate_prompt_union(self):
        out=curate([pair(1,'a','Q'),pair(2,'b','Q'),pair(3,'b','other')])
        nonempty=[v for v in out.values() if v]
        self.assertEqual(len(nonempty),1); self.assertEqual(len(nonempty[0]),2)
    def test_conflicting_duplicate_requires_review(self):
        a=pair(1,'a','Q'); b=pair(2,'b','Q'); b['chosen']='different'
        with self.assertRaises(ValueError): curate([a,b])


if __name__=='__main__': unittest.main()
