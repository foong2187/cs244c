# Deep Fingerprinting Paper Verification and Evaluation Against Modern Defense Mechanisms

This project replicates and extends the research presented in "Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning" (Sirinam et al., 2018).

## Overview

The original study demonstrated that a 1D Convolutional Neural Network (CNN) can autonomously extract complex features from encrypted Tor traffic to identify user activity with over 98% accuracy, effectively bypassing early defenses like WTF-PAD.

## Project Goals

1. Verify the authors' original findings through closed-world replication
2. Extend evaluation by testing the Deep Fingerprinting architecture against contemporary, low-latency defense mechanisms:
   - RegulaTor
   - BRO
3. Determine if the 2018 model remains viable against modern state-of-the-art traffic shaping
4. Evaluate performance in both closed-world classification and realistic open-world scenarios
