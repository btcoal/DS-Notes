# Keynote: Waymo’s Safety Performance

John Scanlong, Staff Research Scientist at Waymo [Waymo's Safety Performance | Keynote Address | 2025 CCAT Global Symposium](https://youtu.be/uzjwq6hyKbA?si=4_HdLye9yfvWB6oA)


**Speaker:** Dr. John Scallan, Staff Safety Researcher, Waymo

**Event:** 2025 CCAT Global Symposium (Center for Connected and Automated Transportation)

**Core Objective:** To detail the multi-layered methodology used to evaluate and validate the safety of an Automated Driving System (ADS) through both prospective (simulative) and retrospective (real-world) lenses.

## I. Foundations: Mission & Technology

Waymo’s goal is to be the **"world’s most trusted driver."** This implies a shift from selling vehicles to developing a "driver" that can be integrated into various platforms.

* **The Hardware Stack:** Currently utilizes the Jaguar I-PACE, featuring a suite of sensors (LiDAR, Camera, Radar) providing 360-degree perception up to 300 meters.  
* **Audio Receptors:** Critical for detecting sirens and emergency vehicle directions before they are visually apparent.  
* **Operational Scale:** As of 2025, Waymo operates commercially in San Francisco, Los Angeles, Phoenix, and Austin, covering hundreds of square miles of complex surface streets 24/7.

* **New Platforms:** Moving toward the Gilly Zeekr and Hyundai Ioniq platforms to lower costs and improve perception range.

## II. The Safety Case Framework

Waymo operates under a **Safety Case**—a formal argument supported by evidence that the system is "safe enough" for its intended use.

* **The Philosophy:** Safety is defined as the **absence of unreasonable risk**.

* **Methodology-Agnostic:** The framework is designed so that the underlying engineering can change as long as the safety evidence remains robust.  
* **The Continuous Cycle:** Safety is not a "one-time check" but an ongoing evaluation loop: Pre-specified targets $\rightarrow$ Evaluation $\rightarrow$ Deployment $\rightarrow$ Monitoring.

**Ref:** [Webb et al. (2020) Waymo’s Safety Methodologies and Safety Readiness Determinations](https://arxiv.org/pdf/2011.00054)

**Ref:** Favarò, F., et al. (2023). *Building a Credible Case for Safety: Waymo’s Approach for the Determination of Absence of Unreasonable Risk.*

## III. Prospective Safety: Counterfactual Simulations

To address the rarity of fatal crashes, Waymo uses "What-If" simulations to reconstruct real-world historical crashes and test the ADS's response.

### The Responder vs. Initiator Methodology

Dr. Scallan’s research involved a census of fatal crashes in Chandler, AZ (2008–2017).

1. **Initiator:** The party making the initial unexpected maneuver (e.g., running a red light).  
   * **Finding:** When the Waymo Driver replaced the initiator, it avoided **100%** of the collisions by simply following rules (e.g., not speeding, stopping at red lights).

2. **Responder:** The party reacting to the initiator’s error.  
   * **Finding:** When the Waymo Driver replaced the responder, it prevented **82%** of crashes and mitigated the severity of another **10%**.

### The "NIEON" Human Benchmark

To avoid comparing the ADS to "distracted or impaired" humans, Waymo developed the **NIEON** model: **Non-Impaired, Eyes-On-the-conflict**.

* **Function:** It models how an *attentive* human would react, considering perception-response time (PRT) and evasive maneuvers (braking vs. steering).  
* **Result:** Waymo generally outperforms even the NIEON model because it perceives 360 degrees simultaneously and never suffers from "startle" delay.

**Ref:** Scanlon, J. M., et al. (2021). *Waymo Simulated Driving Behavior in Reconstructed Fatal Crashes.* 

**Ref:** Engström, J., et al. (2024). *Modeling Road User Response Timing in Naturalistic Traffic Conflicts: A Surprise-Based Framework.*

## IV. Retrospective Safety: Real-World Data

Once the system is deployed, Waymo measures "In-Vivo" (real-world) performance.

### 1\. Insurance Claims Analysis (Collaborated with Swiss Re)

Insurance data provides a "contribution lens"—it shows who was at fault and the financial/physical cost.

* **Property Damage:** 88% reduction in claim frequency compared to humans.

* **Bodily Injury:** 92% reduction in claim frequency compared to humans.

* **Latest Gen. Comparison:** Even when compared only to the newest human-driven cars (2021+), Waymo still showed an \~86% reduction in damage claims.

### 2\. Police Reported Crashes & Spatial Adjustment

Comparing ADS crash rates to national human averages is often "apples-to-oranges" because:

* **Road Type:** Humans drive on freeways (lower risk per mile); Waymo operates mostly on surface streets (higher risk per mile).  
* **Vehicle Type:** Waymo is a passenger platform; human benchmarks often include motorcycles and heavy trucks.  
* **Spatial Bias:** Waymo drives in high-density urban centers where crash rates are naturally higher.  
* **Correction:** Waymo applies a **Spatial Adjustment** to benchmarks, essentially asking: *"How often would a human crash if they drove only on these specific high-risk blocks?"*

**Ref:** Di Lillo, L., et al. (2023). *Waymo’s Safety Performance: A Comparison of ADS and Human-Driven Claims Data.* 

**Ref:** Kusano, K. D., et al. (2024). *Comparison of Waymo Rider-Only Crash Rates by Crash Type to Human Benchmarks.*

## V. External Standards & Transparency

Dr. Scallan emphasizes that safety cannot be a proprietary secret.

* **The RAVE Checklist:** (Retrospective Automated Vehicle Evaluation). A 15-point checklist for researchers to ensure that retrospective studies are valid, transparent, and account for exposure biases.

* **ISO Standardization:** The RAVE checklist has been submitted as a preliminary work item for an **International ISO Standard (ISO/PWI TS 25536)**.

* **Public Data:** Waymo provides a public dashboard where researchers can download mileage, crash data, and heat maps to verify company claims.

## Summary

| Methodology | Purpose | Key Result |
| :---- | :---- | :---- |
| **Safety Case** | Framework for deployment | Proves "Absence of Unreasonable Risk" |
| **Simulations** | Prospective (What-if) | 100% avoidance as initiator; 92% prevention/mitigation as responder |
| **NIEON Model** | Benchmark | Replaces "average" human with "attentive" human |
| **Claims Data** | Retrospective (Financial) | \>90% reduction in bodily injury claims |
| **RAVE Checklist** | Standardize research | Ensures "Apples-to-Apples" comparison |

## References

1. Favarò, F., et al. (2023). *Building a Credible Case for Safety.* Waymo Research.  
2. Scanlon, J. M., et al. (2021). *Waymo Simulated Driving Behavior in Reconstructed Fatal Crashes.* **Peer-reviewed.**
3. Engström, J., et al. (2024). *Modeling Road User Response Timing in Naturalistic Traffic Conflicts.* **Peer-reviewed.**
4. Di Lillo, L., et al. (2023). *Waymo’s Safety Performance: A Comparison of ADS and Human-Driven Claims Data.* (Swiss Re Collaboration).  
5. Scanlon, J. M., et al. (2024/2025). *RAVE Checklist: Recommendations for Overcoming Challenges in Retrospective Safety Studies.*  
