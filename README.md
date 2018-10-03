# -Review-Automatic-Goal-Generation-for-Reinforcement-Learning-Agents
> Carlos Florensa David Held Xinyang Geng Pieter Abbeel  
[Review]
> 참고자료  
> https://www.slideshare.net/DongMinLee32?utm_campaign=profiletracking&utm_medium=sssite&utm_source=ssslideview  
> https://www.mckinsey.com/business-functions/mckinsey-analytics/our-insights/an-executives-guide-to-ai



## [Motivation]
(1) Reinforcement Learning (RL) 은 학습을 통해서 agent 가 작업 (task)를 효과적으로 수행할 수 있도록 하는 강력한 기술입니다.  
그러나, RL을 통해서 학습된 agent를 학습 과정과 같은 "reward function"을 가진 single task 에서만 뛰어난 성능을 보입니다.  
즉, agent가 다양한 작업 (task)들을 수행하는 것이 필요한 상황에서는 효과적이지 못한 모습일 보여줍니다.  
> i.e., 다양한 위치에 존재하는 agent가 다양한 목적지로 이동하는 것과 같은 행동  
> 이는 agent의 현재 위치가 변하면 행할 수 있는 action space가 변하고, 그와 함꼐 변화된 state space에 대해 적절한 action을 선택하지 못하게 되는 것입니다.  

(2) 많은 real-world environment들은 agent가 single task 뿐만 아니라 다양한 작업들을 수행해야하는 경우가 더 많이 존재합니다.   
그러므로, agent 가 가진 current policy 를 가지고 가능한 목표 (possible goals)에 대해서 agent가 평균적인 성공 확률을 최대화 시키는 것에 대한 필요성이 대두되고 있습니다.  


## [Summary]
> 이 논문에서 제안한 알고리즘에 대한 간단한 요약
(1) agent가 자동적으로 현재 agent가 속한 환경에서 수행 할 수있는 작업 (task)의 범위를 발견 할 수있는 방법을 제안합니다.    
(2) Generator neural netwok를 사용하여 agent가 달성하려고 시도하는 작업 (task)를 제안합니다.   
(3) 각 작업 (task)는 state space의 특정 매개 변수화 된 하위 집합에 도달하도록 지정됩니다.   
(4) Generator neural network는 adversarial training을 사용하여 agent의 적절한 난이도에 속하는 작업을 생성함으로써 자동으로 curriculum learning이 이루어지도록 합니다.  

## [Introduction]
1. Reinforcement Learning (RL)은 agent가 state 에 대해서 선택하는 action을 통해서 얻게되는 reward를 최대화 하는 작업을 배우는 알고리즘입니다.   
강화 (Reinforcement)는 동물들이 시행 착오 (trial and error)를 통해서 학습하는 방법 중 하나 입니다.   
> 스키너의 쥐 실험을 예로 들 수 있습니다.  
> ![image](https://user-images.githubusercontent.com/40893452/46404920-8ac88980-c741-11e8-9f0a-d6facf945720.png)
> (1) 굶긴 쥐를 상자에 넣는다.   
> (2) 쥐가 상자 안의 이곳 저곳을 돌아 다닌다.   
> (3) 우연히 지렛대를 누른다.   
> (4) 먹이가 나온다.   
> (5) 지렛대와 먹이의 관계를 처음 본 쥐는 두 대상의 관계에 대해서 알 지 못하고 다른 곳으로 이동한다.   
> (6) 우연히 지렛대를 다시 누르게 된다.   
> (7) 쥐는 5-6 번의 과정을 반복하게 되면서 지렛대와 먹이에 대한 관계를 학습하게 된다.  
> (8) 쥐는 배가 고프면 지렛대를 찾아서 먹이를 얻는 행동을 수행한다.  
> 이때, 이 관계를 알게되는 강화 학습의 중요한 요소는 "보상을 얻게 해주는 행동"의 "빈도의 증가"  
> 증가된 빈도에 따라서 얻게되는 행동으로 두 요소간의 관계를 확고히 한다.  

Machine learning에서의 Reiforcement Learning (RL).   
> (1) 보상(reward)를 통해서 학습  
> (2) 보상은 agent가 선택한 행동 (action)에 대한 환경의 반응  
> (3) 행동 (action)의 결과로 나타나는 보상을 통해서 행동 (action)과 보상 (reward) 에 대한 관계를 학습  
> (4) 보상을 얻게되는 행동을 점점 많이 하도록 학습  

![image](https://user-images.githubusercontent.com/40893452/46404647-c3b42e80-c740-11e8-9a6e-5c7360b03b81.png)

Reinforcement Learning (RL)의 목적은 "최적의 행동 양식 or 정책 (Policy)"를 학습 하는 것  
강화학습은 "순차적"으로 결정을 내려야 하는 문제에 적용이 가능합니다.  
전통적으로 "순차적"으로 내려야 하는 문제를 정의할 때 사용하는 방법 "MDP (Markov Decision Process)".  

2. 최근 바둑, 49 종의 Atari Game, 다양한 robotics 작업 등에 대해서 Reinforcement Learning (RL) 알고리즘은 성공적인 결과를 보여주고 있습니다.  
<p align="center">
  <img src="https://user-images.githubusercontent.com/40893452/46405696-f7dd1e80-c743-11e8-99a7-c1c22c981639.png" width="1000"> </p>
그러나, 위의 agent가 학습하는 환경은 모두 단일 보상 함수 (single reward function)를 가진 단일 작업 (single task)를 최적화 하기위해 학습을 수행하며, 이 학습의 결과로 동일한 단일 작업 (single task)를 수행합니다.  
그러나, 실제 세계 환경에서는 단일 작업이 아닌 다양한 작업을 수행해야하는 agent를 필요로 합니다.  

3. agent가 작동하는 환경 속에서 가능한 목적 (possible goals)에 대한 평균적인 성공 확률을 최대화 시키기 위해서는,  
매 training stage 에서 적절한 목적 (goal)을 선택하는 것이 중요합니다.  
> 현재 agent가 가진 정책 (policy)에 대해서 적절한 수준의 goal을 선택하는 것이 중요합니다.    
이 논문에서 제안하는 알고리즘은 agent가 state space의 subset인 "sub-goal"로 정의되는 문제에 대해 "reward function"을 스스로 생성할 수 있게 합니다.  
이 알고리즘은 GOal Generatvie Adversarial Network (Goal GAN) 으로 정의합니다.  

4.   
GoalGAN 에서 discriminator는 현재 agent의 정책 (policy)를 위한 적절한 수준의 goal인지 아닌지를 평가합니다.  
Generator는 이 이 기준을 충족시키는 Goal을 생성하기 위해서 학습됩니다.  

## [Related Work]



## [Details]
### [Problem Definition]
전통적인 Reinforcement Learning (RL) framework에서 각 타임 스텝 t마다 agent는 action을 선택하고 행동합니다.   
action을 선택하고 행동하는 것은 agent가 하나의 state s(t)에서 s(t+1)로 움직이게 하는 역할을 합니다.  
> 즉, 정책 (policy)는 현재 상태 s(t)에서 행동에 대한 확률 분포 (probability distribution)으로 매핑됩니다.   
이때, agent는 task를 수행할 수 있는 주어진 시간 동안 최대의 reward를 얻을 수 있는 정책을 학습하는 것이 강화학습의 목표 입니다.  
그러나, 학습된 정책은 학습에 활용된 reward function 하나에 대한 최대 기댓값을 얻을 수 있는 정책입니다.  

이 논문에서 제시하는 알고리즘은 ![image](https://user-images.githubusercontent.com/40893452/46410113-a7b98880-c752-11e8-84a0-a77a03c7a861.png) 와 같이 reward function의 range에 대해서도 고려합니다.  

goal g 는 state 의 subset입니다.  ![image](https://user-images.githubusercontent.com/40893452/46410180-de8f9e80-c752-11e8-8a6a-553c07775f74.png)  
그러므로, agent가 s(t)에 goal 에 속해 있다면 목표를 성취했다고 판단하게 됩니다. ![image](https://user-images.githubusercontent.com/40893452/46410381-a0df4580-c753-11e8-8ef5-8e32eb32f377.png)   
이렇게 목표를 성취하는 과정에서 agent는 주어진 goal g 에 대해서 최적화된 정책 (policy)를 학습합니다.  

#### [Simplyfied Reward Function]
state의 subset인 goal g에 대한 학습을 수행하기 위해서는 reward가 필요하며, 이를 위한 reward function으로 다음과 같은  
함수를 정의합니다.  
![image](https://user-images.githubusercontent.com/40893452/46410449-e3a11d80-c753-11e8-8351-0b9a0d13135e.png)

(1) ![image](https://user-images.githubusercontent.com/40893452/46410915-55c63200-c755-11e8-93a5-edd16d2a41e3.png)  
(2) ![image](https://user-images.githubusercontent.com/40893452/46410937-64ace480-c755-11e8-8856-91ce4a4ecdff.png) state s(t)를 goal space에 매핑 시키는 함수  
(3) ![image](https://user-images.githubusercontent.com/40893452/46410964-742c2d80-c755-11e8-81e2-f1b93d108424.png) goal space 내에서의 distance metric  
(4) e (epsilon) :: goal이 도달되었다고 인정 되는 것에 대한 판단에서의 tolerance (acceptable tolerance)

#### [MDP]
매 episode는 agent가 goal state에 도달하였을 때 끝나는 MDP를 정의합니다.  
그러므로, episode의 return은 다음과 같이 정의되며, agent가 최대 T time step 내에 goal에 도달하였는지를 나타내는  
binary random variable이 됩니다.   

![image](https://user-images.githubusercontent.com/40893452/46412303-1a2d6700-c759-11e8-9bb9-e40243327598.png)

> T time step 이내에 agent가 goal에 도달하면 1, 못하면 0 이 되는 return value 이므로, binary random variable이라고 합니다.  

![image](https://user-images.githubusercontent.com/40893452/46412342-3e894380-c759-11e8-998e-d95f2802f929.png)
는 현재 정해진 목표 (current goal) g에 따라 학습됩니다.  






### [Method] 
