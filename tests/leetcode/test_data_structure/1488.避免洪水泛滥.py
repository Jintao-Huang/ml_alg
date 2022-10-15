from libs import *
from libs.alg import *

class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        """将抽的天数存起来. 
        -: lake_date: 对于每一个湖. 最近的下雨的日期. 
            sum_set: 晴天的池, 存日期. 
        若遇到某天下雨, 但是湖以及满了, 则查看上一次下雨和这次下雨之间由于空闲的晴天, 在那一天倒水. 
            若有则清除晴天. 没有则失败.        
        """
        n = len(rains)
        res = [-1] * n
        lake_date = DefaultDict[int, int]()
        sum_date = RBSortedList()
        for date in range(n):
            lake_idx = rains[date]
            if lake_idx == 0:
                sum_date.add(date)
            else:
                if lake_idx not in lake_date:
                    lake_date[lake_idx] = date
                else:
                    last_date = lake_date[lake_idx]
                    i = sum_date.bisect_left(last_date)
                    if i < len(sum_date):  # 存在
                        lake_date[lake_idx] = date
                        d = sum_date[i]
                        res[d] = lake_idx
                        sum_date.pop(i)
                    else:
                        return []
        # 剩下的sum_set, 随意抽光一个湖, 这里取第一个湖
        for date in sum_date:
            res[date] = 1
        return res

if __name__ =="__main__":
    rains = [1,0,2,0,3,0,2,0,0,0,1,2,3]
    print(Solution().avoidFlood(rains)  ) 
