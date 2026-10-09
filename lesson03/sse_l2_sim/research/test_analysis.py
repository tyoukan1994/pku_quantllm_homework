import unittest
import numpy as np
from analyze import FastBook, event_ofi, feature_matrix, target, partitions, transform_fit, transform, rolling


def addition(oid,side,price,qty):
    return dict(BizIndex=oid,OrderOriNo=oid,FunctionCode=side,Price=price,Volume=qty,OrderKind=65,Time=93000000)


class ResearchChecks(unittest.TestCase):
    def test_fast_best_tracks_level_depletion_and_reappearance(self):
        b=FastBook()
        for x in [addition(1,66,10,100),addition(2,66,9.99,200),addition(3,83,10.01,300)]:b.apply('O',x)
        self.assertEqual(b.quote(),(1000,100,1001,300))
        x=addition(1,66,10,100);x['OrderKind']=68;b.apply('O',x)
        self.assertEqual(b.best(66),(999,200))
        b.apply('O',addition(4,66,10,50));self.assertEqual(b.best(66),(1000,50))

    def test_event_ofi_sign_and_price_change_rules(self):
        old=(1000,100,1001,200)
        self.assertEqual(event_ofi(old,(1000,150,1001,200)),50)
        self.assertEqual(event_ofi(old,(1000,100,1001,250)),-50)
        self.assertEqual(event_ofi(old,(1001,70,1002,90)),270)
        self.assertEqual(event_ofi(old,(999,70,1000,90)),-190)

    def test_rolling_window_excludes_older_and_future_events(self):
        np.testing.assert_equal(rolling(np.array([1,2,3,4,5]),3),[1,3,6,9,12])

    def test_all_features_ignore_later_mutations(self):
        raw=np.ones((12,19));raw[:,0]=np.arange(34200,34212);raw[:,1]=0
        raw[:,2]=1000;raw[:,3]=1002;raw[:,4:10]=100;raw[:,18]=0
        before,_=feature_matrix(raw)
        changed=raw.copy();changed[8:,2:18]*=100
        after,_=feature_matrix(changed)
        np.testing.assert_equal(before[:8],after[:8])

    def test_labels_do_not_cross_lunch_or_data_end(self):
        raw=np.zeros((8,19));raw[:,0]=[41397,41398,41399,41400,46800,46801,46802,46803]
        raw[:,1]=[0,0,0,0,1,1,1,1];raw[:,2]=1000;raw[:,3]=1002
        mid=np.arange(1001,1009);y,ok,_=target(raw,mid,2)
        np.testing.assert_equal(ok,[True,True,False,False,True,True,False,False])
        self.assertTrue(np.isnan(y[2]))

    def test_validation_and_test_labels_are_purged_at_split_boundaries(self):
        raw=np.zeros((6,19));raw[:,0]=[41100,41101,50100,50101,53520,53521];raw[:,1]=[0,0,1,1,1,1]
        train,val,test=partitions(raw,300,np.ones(6,dtype=bool))
        np.testing.assert_equal(train,[True,False,False,False,False,False])
        np.testing.assert_equal(val,[False,False,True,False,False,False])
        np.testing.assert_equal(test,[False,False,False,False,True,False])

    def test_scaling_fitted_on_training_only(self):
        train=np.arange(100).reshape(-1,1).astype(float);params=transform_fit(train)
        expected=transform(train,params).copy();test=np.array([[1000000.]])
        transform(test,params)
        np.testing.assert_equal(transform(train,params),expected)
        self.assertLess(float(transform(test,params)[0,0]),2)


if __name__=='__main__':unittest.main()
