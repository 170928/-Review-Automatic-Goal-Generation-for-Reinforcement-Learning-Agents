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
다중 목적을 위한 정책을 학습하는 강화학습은 다음과 같이 2가지로써 불린다.  
(1) multi-task policy search (Deisenroth, 2014)  
(2) contextual policy search (Deisenroth, 2013 & Fabisch, 2014)  

이 논문에서 제안하는 알고리즘은 위의 논문들과는 다음과 같은 다른 특징을 가진다.  
(1) Sparse reward 환경에서도 효과적으로 문제를 해결 가능하도록 해주며, curriculum 방식을 통해서 효과적으로 multi-task를 수행할 수 있도록 해준다.  
(2) 위의 논문들은 정해진 적은 수의 contexts/ tasks들이 학습에 사용되지만, 제안하는 알고리즘은 "contunuous task space"에서 직접적으로 curriculum을 위한 task/context를 만들어낸다.  

### [Intrinsic Motivation]
> 내재적 동기  
> extrinsic motivation (외적 동기)는 생물체의 생존과 번식과 관련된 기본적 요구를 충족시키기위한 행동 습득을 유도하도록 유도된다.   
> 내재적 동기는 내재적 동기가 없는 지식 (예 > 예측 능력)과 역량 (즉, 수행 능력)을 획득하는 진화 적 기능을 수행하는 동기이다.    
> Reinforcement Learning에서는 수행 능력 (objective function)을 최대화 하고자하는 목적을 위한 수행 능력을 획득하기 위한 기능을 수행하게하는  
> 동기를 의미.  

------------------------------------------------------------------------------------------------------------------

#### [추가 필수 개념]
(1) exploration, the agent experiments with novel strategies that may improve returns in the long run  
(2) exploitation, it maximizes rewards through behavior that is known to be successful  
> An effective exploration strategy allows the agent to generate trajectories that are maximally informative about the environment
그러나 exploration 과정은 small space에서 효과적으로 작동하며, continuous space에서는 사용되기 어려웠습니다.  
그 결과, e-greedy search/ Gaussian Noise 등과 같은 방법들이 적용되어서 exploration이 high dimensional space에서도 이루어질 수 있도록 연구가 진행되었습니다.    
하지만, 이러한 exploration은 학습시에 환경의 크기에 따라서 학습 시간을 증가시킵니다....   
그리하여, "surprise" 라는 개념 "intrinsic" 이라는 개념을 등장시키게 된 연구가 존재합니다.  
이 개념은 surprise를 이끄는 action을 agent가 수행하게 한다 라는 개념으로 surprise의 의미는 Dynamic model distriubtion을 크게 변화시키는 action을 선택한다는 것을 의미합니다. (새로운 정보를 많이 얻을 수 있는 action을 취한다)   

-----------------------------------------------------------------------------------------------------------------

Intrinsic Motivaton 개념은 policy 학습과정에서 state space를 탐사하는 방법에 대한 연구 분야를 의미합니다.  
기존의 RL에서의 e-greedy search / Gaussian Noise based search 들을 생각하시면 됩니다.  
이 논문에서 제안하는 방법은 이전 연구들에서 제시한 알고리즘들 보다도 효과적이며 sparse reward environment에서도 효과적으로 학습이 이루어질 수 있도록 합니다.  
개념적인 차이로는 intrinsic motivation이 agent가 학습함에 있어서 학습이 어렵게 만드는 state에 대한 개념을 이용 (고려) 하지 않는다는 것입니다.
> 중간 state에서 여러 곳으로 action을 취할 수 있어 해당 state로 부터 학습이 어려워지는 포인트가 있다면.... 예로 도시의 광장과 같이 여러길을 가진 그러나 모두 어떻게든 목적에 도달할수 있는 state라고 생가하면 될 것 같습니다.   
제시하는 알고리즘은 직접적으로 agent가 학습과정에서 학습을 어렵게 만드는 task를 지정하고 학습을 수행하도록 합니다.  

### [Skill-learning]
skill-learning은 하나의 agent가 여러가지 task를 수행 할 수 있도록 하기 위한 방법에 대한 연구 중 하나의 분야입니다.  
agent가 scratch (처음) 부터 학습을 모두 하는 것이 아니라 기존에 학습한 skill을 재사용하여 학습시에 improve를 수행하도록 하는 개념입니다.  
그러나, useful한 skill이란 것을 정희하고 pre-training 하는 것은 매우 어려운 문제입니다.  

제시하는 알고리즘은 skill 같은 요소를 사전에 찾고 pretraining 하는 과정 없이, multi-task 를 위해서 직접적으로 정책를 학습하는 방법을 통해서 문제를 해결합니다.  

### [Curriculum Learning]
multi-task를 위한 방법중의 하나로 skill-learning과 다른 방법 중 하나로 curriculum learning이 존재합니다.  
간단하게 이해하자면, agent의 학습 과정에서 task를 최적화 시켜서 agent가 단계적으로 학습해 나갈 수 있게하는 방법을 의미합니다.  
문제는 이러한 단계적 학습 대상이 "hand-crafted task"입니다.  
그러므로, pre-specified sequence of task에 의존해서 학습하게 되는 한계점이 존재합니다.  

이 논문에서 제안하는 알고리즘은 지속적으로 task를 일반화하는 policy를 훈련하며 현재의 agent 성능에 너무 어려운 작업 (goal)에 training effort를 과다하게 할당하지 않아 sparse reward 환경에서도 효과적으로 작동합니다.  

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

(1) episode의 return은 다음과 같이 정의되며,   

![image](https://user-images.githubusercontent.com/40893452/46412303-1a2d6700-c759-11e8-9bb9-e40243327598.png)

agent가 최대 T time step 내에 goal에 도달하였는지를 나타내는 binary random variable이 됩니다.   
> T time step 이내에 agent가 goal에 도달하면 1, 못하면 0 이 되는 return value 이므로, binary random variable이라고 합니다.  

(2)  
![image](https://user-images.githubusercontent.com/40893452/46412342-3e894380-c759-11e8-998e-d95f2802f929.png)
는 현재 정해진 목표 (current goal) g에 따라 다르게 학습이 이루어집니다.  
> g 가 prior 의 개념으로 포함된다는 것을 의미합니다.  
> 즉, 같은 state상황에서도 g 에 따라 action에대한 확률 분포 (정책)이 변합니다.  

(3) current policy를 통해서 MDP를 따라갈 때 얻게되는 샘플링에서 얻게되는 기대 수익 (expected return)은  
T time-step 내에 해당 목표 g 에 대한 성공 확률로 나타냅니다.  
![image](https://user-images.githubusercontent.com/40893452/46425549-e5c8a380-c776-11e8-81a8-872d8c091313.png)

(4) (1)에서의 Return 식과 (3) 에서의 Return에 대한 식은 다른 특징을 가집니다.   
(1)에서의 Return 식은 "sparse indicator reward function" 으로 simple 하면서 실제로 real-wrold goal problem을 잘 표현합니다.  
> 미로 찾기와 같이 골을 달성하고 나면 성공했다고 말을할 수 있지만, 미로 찾기 도중에는 잘 가고있다고 알 수 없는 실제문제  
> 이론적으로, 프로그래머가 직접 "meaningful distance function"을 만들어서 "dense reward function"을 형성할 수 있습니다.  
> 이는 미로의 특징을 찾아서 미로의 도입부는 배경이 빨간색, 중간은 파란색, 마지막은 초록색 과 같은 특징에 따라 파란색이 보이면 중간 reward를 제공하는 것과 같은 예로 이해할 수 있습니다.  

(5) 제안되는 방법은 (1) / (3) 수식과 같은 Indicator로써 표현되는 binary random variable return에 대해서도 학습이 가능합니다.  

#### [Objective] 
many goals "g"에 대해서 high reward를 받을 수 있게 되는 
![image](https://user-images.githubusercontent.com/40893452/46569831-c2c80a80-c995-11e8-927f-bb1e0496aa6b.png) 를 찾는것이 목적이 됩니다.  
수식으로는 다음과 같이 표현할 수 있습니다.  
![image](https://user-images.githubusercontent.com/40893452/46569827-adeb7700-c995-11e8-8939-c20402c5d334.png)

R(.)는 앞서 설명했듯이 goal g 에 대한 success probability를 의미합니다.  
그러므로, 위의 식은 p_g(g)의 확률 분포를 따라서 G set에서 sampled 된 goals g 에 대한 정책이 갖는 평균적인 성공 확률을 의미합니다.  

#### [Assumption]
(1) 목표 공간의 일부 영역에서 충분한 수의 목표에 대해 학습 된 정책 (policy)는 해당 영역 내의 다른 목표로 보간하는 법을 배웁니다.  
(2) 일부 목표에 대해 학습 된 정책 (policy)는 학습에 사용된 목표와 유사한 목표에 대해 학습할 때 좋은 "초기화" 요건이 됩니다. 
> 기반이 되기 좋은 정책이라는 것을 의미합니다.  
> transfer learning을 생각하면 됩니다.  
즉, 학습된 정책은 목표에 간혹 도달 할 수는 있지만 일관성 (항상 도달함)을 의미하지는 않는다는 것입니다.
 

### [Method] 
다음과 같은 3단계의 알고리즘이 존재합니다.  
(1) 현재 정책에 대해서 적당한 난이도를 가진 목표인지 아닌지를 기반으로 목표의 set에 대해 label을 할당  
(2) label을 기반으로ㅡ generator를 학습하여 적절한 난이도의 새로운 목표를 생성할 수 있도록 한다.  
(3) 정책을 효과적으로 학습하기 위해서 위에서 만들어진 새로운 목표를 사용합니다.  

#### [Goal Labelling]

#### [Adversarial Goal Generation]

#### [Policy Optimization]


