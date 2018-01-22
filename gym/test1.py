import gym
env = gym.make('MountainCar-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    
    
异常值检测

1，极值分析
通过scatterplots,histogramas, box和whisker plot分析极值。
查看样本分布（假设高斯分布），去距离1/4和3/4值2-3倍标准差数值的样本。
2，临近方法
基于k-means分析样本质心，去掉离质心特别远的样本。
3，投影方法
通过PCA，SOM，sammon mapping去掉不重要特征。