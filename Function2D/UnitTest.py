import unittest
from Function2D.Main import GetMSEofApproximation, GetDomainAndFunctions


_, __, Original_Function, Function_By_NN = GetDomainAndFunctions()


class MyTests(unittest.TestCase):


    def test1(self):
        self.assertTrue(GetMSEofApproximation(Original_Function,Function_By_NN) < 20)
    def test2(self):
        self.assertTrue(GetMSEofApproximation(Original_Function, Function_By_NN) < 10)
    def test3(self):
        self.assertTrue(GetMSEofApproximation(Original_Function, Function_By_NN) < 5)

if __name__=="__main__":
    unittest.main()