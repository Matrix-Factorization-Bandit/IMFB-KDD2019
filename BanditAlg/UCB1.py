class UCB1Algorithm:
    def __init__(self):
        self.arms = {}
        self.TotalPlayCounter = 0
        
    def decide(self, arm_pools):
        self.TotalPlayCounter +=1
        arm_Picked = None
        maxPTA = float('-inf')
        
        for x in arm_pools:
            if x.id not in self.arms:
                self.arms[x.id] = UCB1Struct(x.id)

            if self.arms[x.id].numPlayed == 0:
                article_Picked = x
                return article_Picked

            x_pta = self.arms[x.id].getProb(self.TotalPlayCounter)
            if maxPTA < x_pta:
                article_Picked = x
                maxPTA = x_pta
        return article_Picked       
         
    def updateParameters(self, arm, click): 
        self.articles[arms.id].updateParameters(click)
