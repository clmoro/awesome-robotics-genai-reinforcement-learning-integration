# Robotics-RL-FMs-Integration
This repository contains a curated list of the papers classified in the survey titled ***"Integrating Reinforcement Learning with Foundation Models for Autonomous Robotics: Methods and Perspectives"***. We also provide five Excel files (one for each category) that offer detailed summaries of the analyses we performed using the paper's taxonomy. These summaries cover several features of the analyzed papers, such as `name of the framework`, `model used`, `code availability`, `dataset`, `type of application`, `simulation vs. real-world`, `subcategories`, `experiment evaluation`, `year of publication`, `RL for FM vs. FM for RL`, and `short description`.
## Abstract
Large pre-trained models, such as foundation models (FMs), despite their powerful abilities to understand complex patterns and generate sophisticated outputs, often struggle with adapting to specific tasks. Reinforcement learning (RL), which allows agents to learn through interaction and feedback, presents a compelling solution. Integrating RL empowers foundation models to achieve desired outcomes and excel at specific tasks. Simultaneously, RL itself can be enhanced when coupled with the reasoning and generalization capabilities of FMs. The synergy between foundation models and RL is revolutionizing many fields, robotics is among them. Foundation models, rich in knowledge and generalization capabilities, provide robots with a wealth of information, while RL enables them to learn and adapt through real-world interaction. This survey paper offers a comprehensive exploration of this exciting intersection, examining how these paradigms can be integrated to push the boundaries of robotic intelligence. We analyze the use of foundation models as action planners, the development of robotics-specific foundation models, and the mutual benefits of combining foundation models with RL. We also present a taxonomy of integration approaches, including large language models, vision-language models, diffusion models, and transformer-based RL models. Finally, we delve into how RL can harness the world representations learned from foundation models to enhance robotic task execution. Through synthesizing current research and highlighting key challenges, this survey aims to spark future research and contribute to the development of more intelligent, adaptable, and capable robotic systems. To summarize the analysis conducted in this work, we also provide a continuously updated collection of papers based on our taxonomy.
## 1. Large Language Models Enhance Reasoning Capabilities in RL Agents
 ### 1.1 Inverse RL: generating the reward function through LLMs
  - Accelerating Reinforcement Learning of Robotic Manipulations via Feedback from Large Language Models [link](https://arxiv.org/abs/2311.02379)
  - Augmenting Autotelic Agents with Large Language Models
  - Eureka: Human-Level Reward Design via Coding Large Language Models
  - FoMo Rewards: Can we cast foundation models as reward functions?
  - Guiding Pretraining in Reinforcement Learning with Large Language Models
  - Intrinsic Language-Guided Exploration for Complex Long-Horizon Robotic Manipulation Tasks
  - Language as a Cognitive Tool to Imagine Goals in Curiosity-Driven Exploration
  - Language to Rewards for Robotic Skill Synthesis
  - Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation
  - Self-Refined Large Language Model as Automated Reward Function Designer for Deep Reinforcement Learning in Robotics
  - Text2Reward: Reward Shaping with Language Models for Reinforcement Learning
 ### 1.2 Large language models to directly generate or refine RL policies
  - Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance
  - Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning
  - Language Instructed Reinforcement Learning for Human-AI Coordination
 ### 1.3 Grounding LLM plans in real world through RL generated primitives
  - Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
  - Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents
  - Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks
  - Prompt, Plan, Perform: LLM-based Humanoid Control via Quantized Imitation Learning
## 2. Vision Language Models for RL-Based Decision Making
  - Can Foundation Models Perform Zero-Shot Task Specification For Robot Manipulation?
  - Code as Reward: Empowering Reinforcement Learning with VLMs
  - Foundation Models in Robotics: Applications, Challenges, and the Future
  - Language Reward Modulation for Pretraining Reinforcement Learning
  - LIV: Language-Image Representations and Rewards for Robotic Control
  - RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback
  - RoboCLIP: One Demonstration is Enough to Learn Robot Policies
  - Robot Fine-Tuning Made Easy: Pre-Training Rewards and Policies for Autonomous Real-World Reinforcement Learning
  - Towards A Unified Agent with Foundation Models
  - Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning
  - Vision-Language Models as a Source of Rewards
  - Vision-Language Models Provide Promptable Representations for Reinforcement Learning
  - Zero-Shot Reward Specification via Grounded Natural Language
  - ExploRLLM: Guiding Exploration in Reinforcement Learning with Large Language Models
  - Affordance-Guided Reinforcement Learning via Visual Prompting
## 3. RL Robot Control Empowered by Diffusion Models
 ### 3.1 Diffusion models for policy generation and representation
  - Consistency Models as a Rich and Efficient Policy Class for Reinforcement Learning
  - Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning
  - Beyond Conservatism: Diffusion Policies in Offline Multi-agent Reinforcement Learning
  - Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
  - Generating Behaviorally Diverse Policies with Latent Diffusion Models
  - Hierarchical Diffusion for Offline Decision Making
  - Is Conditional Generative Modeling All You Need for Decision-Making?
  - Policy Representation via Diffusion Probability Model for Reinforcement Learning
  - Offline Skill Diffusion for Robust Cross-Domain Policy Learning
  - Score Regularized Policy Optimization through Diffusion Behavior for Efficient Offline Reinforcement Learning
  - Policy-Guided Diffusion
 ### 3.2 Diffusion models for planning
  - AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners
  - Adaptive Online Replanning with Diffusion Models
  - Cold Diffusion on the Replay Buffer: Learning to Plan from Known Good States
  - Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning
  - DiPPeR: Diffusion-based 2D Path Planner applied on Legged Robots
  - EDGI: Equivariant Diffusion for Planning with Embodied Agents
  - Hierarchical Diffuser: Efficient Hierarchical Planning with Diffusion Models for Improved Long-Horizon Decision-Making
  - Language Control Diffusion: Efficiently Scaling Through Space, Time, and Tasks
  - Planning with Diffusion for Flexible Behavior Synthesis
  - Refining Diffusion Planner for Reliable Behavior Synthesis by Automatic Detection of Infeasible Plans
  - SafeDiffuser: Safe Planning with Diffusion Probabilistic Models via Control Barrier Functions
  - SSD: Sub-trajectory Stitching with Diffusion Model for Goal-Conditioned Offline Reinforcement Learning
  - Simple Hierarchical Planning with Diffusion
 ### 3.3 Diffusion models for offline RL
  - Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning
  - Efficient Diffusion Policies for Offline Reinforcement Learning
  - Fighting Uncertainty with Gradients: Offline Reinforcement Learning via Diffusion Score Matching
  - IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies
  - Instructed Diffuser with Temporal Condition Guidance for Offline Reinforcement Learning
  - Learning a Diffusion Model Policy from Rewards via Q-Score Matching
  - Learning to Reach Goals via Diffusion
  - MADIFF: Offline Multi-agent Learning with Diffusion Models
  - MetaDiffuser: Diffusion Model as Conditional Planner for Offline Meta-RL
  - Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling
  - Reasoning with Latent Diffusion in Offline Reinforcement Learning
 ### 3.4 Diffusion models for inverse RL
  - Extracting Reward Functions from Diffusion Models
  - Reward-Directed Conditional Diffusion Models for Directed Generation and Representation Learning
  - Diffused Value Function: Value Function Estimation using Conditional Diffusion Models for Control
  - Diffusion Reward: Learning Rewards via Conditional Video Diffusion
## 4. Reinforcement Learning Leverages Video Prediction and World Models
 ### 4.1 Learning robotic tasks with video prediction
  - Foundation Reinforcement Learning (FRL)
  - Learning Generalizable Robotic Reward Functions from 'In-The-Wild' Human Videos
  - Video prediction models as rewards for reinforcement learning
  - Learning reward functions for robotic manipulation by observing humans
  - Vip: Towards universal visual reward and representation via value-implicit pre-training
  - Learning Universal Policies via Text-Guided Video Generation
  - Robotic offline rl from internet videos via value-function pre-training
  - Where are we in the search for an artificial visual cortex for embodied intelligence?
 ### 4.2 Foundation world models for model-based RL
  - Masked World Models for Visual Control
  - Multi-View Masked World Models for Visual Robotic Manipulation
  - Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling
  - EnvGen: Generating and Adapting Environments via LLMs for Training Embodied Agents
  - UniSim: Learning Interactive Real-World Simulators
  - RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation
  - Recurrent World Models Facilitate Policy Evolution
  - GenSim: Generating Robotic Simulation Tasks via Large Language Models
  - GenRL: Multimodal Foundation World Models for Generalist Embodied Agents
  - iVideoGPT: Interactive VideoGPTs are Scalable World Models
  - Zero-shot Safety Prediction for Autonomous Robots with Foundation World Models
  - Genie: Generative Interactive Environments
## 5. Transformer Reinforcement Learning Models
  - Multi-agent reinforcement learning is a sequence modeling problem
  - Hyper-decision transformer for efficient online policy adaptation
  - Prompt-tuning decision transformer with preference ranking
  - Pre-training for robots: Offline rl enables learning new tasks from a handful of trials
  - Think before you act: Unified policy for interleaving language reasoning with actions
  - Online Foundation Model Selection in Robotics
  - Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem
  - A generalist agent
  - HarmoDT: Harmony Multi-Task Decision Transformer for Offline Reinforcement Learning
  - Transformers are adaptable task planners
  - Pact: Perception-action causal transformer for autoregressive robotics pre-training
  - Latte: Language trajectory transformer
  - Q-transformer: Scalable offline reinforcement learning via autoregressive q-functions
  - Anymorph: Learning transferable polices by inferring agent morphology
## Citation

If you find our project useful, please cite our paper:

```

```
