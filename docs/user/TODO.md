# robot-lab Research TODO

**Project Goal**: Build deep RL/robotics expertise through systematic curriculum learning and sim2real research, leading to top company positions and potential scientific contributions.

**Timeline**: 44 weeks (Feb 2026 - Dec 2026)

---

## Phase 0: Core Foundations (Weeks 1-4) - **IN PROGRESS**

### ✅ COMPLETED
- [x] Set up Stable-Baselines3 training pipeline
- [x] Implement multi-environment vectorization
- [x] Create JSON-based config system
- [x] Build experiment tracking infrastructure
- [x] Add TensorBoard integration
- [x] Implement custom environments (A1Quadruped, Gripper, Pusher)
- [x] Set up CLI with Typer
- [x] Add metadata tracking (git, system info, hyperparameters)

### 🎯 CURRENT FOCUS

#### Goal 1: SAC vs PPO Comparison Document
**SMART Goal**: Create a 5-10 page comparison document covering theory and empirical results by **Week 4 (March 2026)**

**Specific Tasks**:
- [ ] **Theory Section** (4 hours)
  - [ ] Explain SAC mathematical foundations (soft Q-learning, entropy maximization)
  - [ ] Explain PPO mathematical foundations (clipped surrogate objective)
  - [ ] Create decision tree: when to use SAC vs PPO
  - [ ] List failure modes and debugging strategies
  
- [ ] **Empirical Comparison** (12 hours)
  - [ ] Run SAC on 3 environments (MountainCar, Walker2d, HalfCheetah) - 5 seeds each
  - [ ] Run PPO on same 3 environments - 5 seeds each
  - [ ] Compare: sample efficiency, final performance, training stability
  - [ ] Hyperparameter sensitivity analysis (learning rate, batch size)
  
- [ ] **Documentation** (4 hours)
  - [ ] Write comparison document with plots and tables
  - [ ] Add code examples showing usage
  - [ ] Include troubleshooting guide
  
**Success Criteria**:
- Document is interview-ready (can explain every section)
- Empirical results are reproducible (all configs saved)
- Plots show clear trends with confidence intervals

---

#### Goal 2: CSV Export & Analysis Tools
**SMART Goal**: Implement CSV export and statistical analysis utilities by **Week 4**

**Specific Tasks**:
- [ ] Add CSV export to ExperimentTracker (2 hours)
- [ ] Create plotting script for learning curves with variance bands (3 hours)
- [ ] Implement statistical significance tests (t-test, bootstrap) (4 hours)
- [ ] Write example analysis notebook (3 hours)

**Success Criteria**:
- Can export any experiment to CSV with one command
- Can generate publication-quality plots
- Can determine if improvement is statistically significant

---

## Phase 0.5: Trajectory Generation (Weeks 4-6) - **LEARNING CRITICAL**

### 🧠 Pre-Implementation Discussion Required
Before implementing, you must be able to explain:
1. Why raw action outputs cause jerky movement
2. Different trajectory parameterization methods (splines, minimum jerk, etc.)
3. Tradeoffs between smoothness and task performance
4. How action filtering affects learning dynamics

### 🎯 Goals

#### Goal 3: Trajectory Generation Framework
**SMART Goal**: Implement and compare 3 trajectory smoothing approaches by **Week 6 (March 2026)**

**Specific Tasks**:
- [ ] **Research & Discussion** (4 hours)
  - [ ] Read papers on robot trajectory generation
  - [ ] Discuss with AI: trajectory parameterization approaches
  - [ ] Design experiment comparing raw vs smoothed actions
  
- [ ] **Implementation** (16 hours) - **YOU IMPLEMENT**
  - [ ] Implement action smoothing filter (low-pass, EMA)
  - [ ] Add temporal regularization loss (action change penalty)
  - [ ] Create trajectory primitive library
  - [ ] Implement wrapper for action filtering
  
- [ ] **Evaluation** (8 hours)
  - [ ] Train baseline (raw actions) on Walker2d
  - [ ] Train smoothed variants (3 approaches)
  - [ ] Measure smoothness metrics (jerk, action variance)
  - [ ] Record videos for qualitative comparison
  
- [ ] **Documentation** (4 hours)
  - [ ] Write trajectory generation guide
  - [ ] Document when to use each approach
  - [ ] Create comparison table and videos

**Success Criteria**:
- Can articulate why each approach works
- Smoothed policies show <50% jerk of baseline
- Task performance maintained (>90% of baseline reward)
- Videos show visually smoother motion

---

## Phase 1: Curriculum Learning Basics (Weeks 7-12)

### 🧠 Pre-Implementation Discussion Required
Before implementing curriculum logic, you must explain:
1. What makes a good curriculum (not too easy, not too hard)
2. How to detect when agent is ready to advance
3. Catastrophic forgetting and how to prevent it
4. Curriculum vs domain randomization tradeoffs

### 🎯 Goals

#### Goal 4: Manual Curriculum Infrastructure
**SMART Goal**: Build curriculum framework and run first comparison by **Week 9 (April 2026)**

**Specific Tasks**:
- [ ] **Infrastructure** (AI can help, 8 hours)
  - [ ] Create `CurriculumCallback` base class with logging
  - [ ] Add curriculum state tracking to ExperimentTracker
  - [ ] Implement environment difficulty wrapper
  - [ ] Build multi-stage experiment runner
  
- [ ] **Curriculum Design** (YOU design, 12 hours)
  - [ ] Design 4-stage terrain curriculum (Flat → Friction → Obstacles → Climbing)
  - [ ] Define advancement criteria (reward threshold? episode count?)
  - [ ] Implement curriculum scheduling logic
  
- [ ] **Experiment** (12 hours)
  - [ ] Train baseline (hardest level only) - 5 seeds
  - [ ] Train curriculum (4 stages) - 5 seeds
  - [ ] Compare timesteps to success, stability, final performance
  - [ ] Measure state visitation entropy at each stage
  
**Success Criteria**:
- Curriculum shows faster convergence than baseline
- Results are statistically significant (p < 0.05)
- Can explain when/why curriculum helps

---

#### Goal 5: Adaptive Curriculum Advancement
**SMART Goal**: Implement and compare 3 advancement strategies by **Week 12 (May 2026)**

**Specific Tasks**:
- [ ] **Design** (YOU design, 8 hours)
  - [ ] Fixed episode schedule
  - [ ] Reward-threshold based
  - [ ] Regression-based (move back on failure)
  
- [ ] **Implementation** (YOU implement, 12 hours)
  - [ ] Implement advancement logic for each strategy
  - [ ] Add metrics tracking (advancement events, regression count)
  
- [ ] **Experiment** (16 hours)
  - [ ] Run all 3 strategies on 2 environments
  - [ ] Compare convergence speed, stability, final performance
  - [ ] Measure catastrophic forgetting events

**Success Criteria**:
- Can explain pros/cons of each strategy
- Have data-driven recommendation for which to use
- Results inform Phase 2 experiments

---

## Phase 2: Representation & Generalization (Weeks 13-20)

### 🎯 Goals

#### Goal 6: Goal-Conditioned Curriculum
**SMART Goal**: Implement goal-conditioned learning and test generalization by **Week 16 (June 2026)**

**Specific Tasks**:
- [ ] Study goal-conditioned RL (HER, goal representations)
- [ ] Implement goal-augmented observation space (YOU implement)
- [ ] Design curriculum with varying goal positions
- [ ] Test zero-shot generalization to unseen goals

**Success Criteria**:
- >60% success on unseen goal configurations
- Can explain how goal representation aids generalization

---

#### Goal 7: Curriculum + Domain Randomization
**SMART Goal**: Study interaction between curriculum and randomization by **Week 20 (July 2026)**

**Specific Tasks**:
- [ ] Implement domain randomization (friction, mass, motor strength)
- [ ] Run factorial experiment (curriculum × randomization)
- [ ] Measure robustness across perturbations

**Success Criteria**:
- Understand optimal balance between structured and random learning
- Have actionable insights for Phase 3 transfer experiments

---

## Phase 3: Sim2Real Research (Weeks 21-32)

### 🎯 Goals

#### Goal 8: Morphology Transfer
**SMART Goal**: Study curriculum's effect on transfer learning by **Week 26 (August 2026)**

**Specific Tasks**:
- [ ] Create morphology variants (leg length, mass, actuator strength)
- [ ] Compare: scratch training vs curriculum pretrain + finetune
- [ ] Measure zero-shot transfer success

---

#### Goal 9: Dynamics Gap Sensitivity
**SMART Goal**: Quantify curriculum's robustness benefits by **Week 32 (October 2026)**

**Specific Tasks**:
- [ ] Simulate dynamics gaps (latency, contact noise, perturbations)
- [ ] Measure performance degradation curves
- [ ] Compare curriculum vs baseline robustness

---

## Phase 4: Scientific Contribution (Weeks 33-44)

### 🎯 Goals

#### Goal 10: Contribution Paper Submission
**SMART Goal**: Submit workshop or conference paper by **Week 44 (December 2026)**

**Specific Tasks**:
- [ ] Choose contribution direction (exploration shaping, curriculum/DR tradeoff, or transfer-optimal curriculum)
- [ ] Run comprehensive experiments (30+ seeds)
- [ ] Write 4-8 page paper
- [ ] Submit to RL conference workshop

**Success Criteria**:
- Novel insight supported by rigorous experiments
- Publication-quality plots and analysis
- Accepted submission (stretch goal)

---

## Evaluation Criteria

### Weekly Progress Metrics
- **Learning Hours**: 15-20 hours/week (theory + implementation + experiments)
- **Experiments Completed**: 2-4 per phase
- **Documentation**: 1 markdown doc per major experiment
- **Code Quality**: All PRs pass linting, have tests

### Phase Completion Criteria
Each phase is complete when:
1. All SMART goals met with success criteria
2. Results are reproducible (configs saved, seeds logged)
3. Documentation written (what/why/insights)
4. Next phase has clear starting point

### Overall Success Indicators
- **Month 3**: SAC/PPO expert, trajectory generation working
- **Month 6**: First curriculum learning paper insights
- **Month 9**: Transfer learning experiments complete
- **Month 11**: Paper submitted, portfolio ready

---

## Intermediary Steps & Risk Mitigation

### Potential Bottlenecks
1. **GPU access**: GTX 1080 may be slow for large experiments
   - **Mitigation**: Start with simple envs, parallelize training
2. **Experiment time**: Full runs may take days
   - **Mitigation**: Use quick smoke tests, then scale up
3. **Statistical significance**: Need many seeds
   - **Mitigation**: Budget 5-10 seeds per experiment, use efficient variance reduction

### Checkpoints for Adjustment
- **Week 6**: If trajectory generation is too hard, defer to Phase 1
- **Week 12**: If curriculum shows no benefit, pivot to other research questions
- **Week 20**: Reassess scientific contribution direction based on Phase 1-2 results
- **Week 32**: Decide paper topic and submission target

---

## Notes
- **Learning Priority**: Understanding > Speed. Take time to deeply grasp concepts.
- **Documentation First**: Write "what I learned" docs immediately after experiments.
- **Ask for Help**: Use Socratic discussion with AI to clarify concepts before implementing.
- **Celebrate Wins**: Each completed SMART goal is progress toward job/research goals.
