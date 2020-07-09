Copyright (C) 2020 ETH Zurich, Switzerland. 
SPDX-License-Identifier: GPL-3.0-only.
See LICENSE file for details.

Author: Edoardo Mello Rella <edoardom@student.ethz.ch>

# C Implementation

Containing C implementations of the regression methods discussed.
Every method is designed for performance testing purpose and input features are considered to be precomputed at earlier stages.

# Usage

First install [Pulp-SDK](https://github.com/pulp-platform/pulp-sdk).
Tests were done on the release [2019.04.05-pulp-sdk-ubuntu-16](https://github.com/pulp-platform/pulp-sdk/releases/tag/2019.04.05).

Configure properly dependencies and sources executing:

```
export PULP_RISCV_GCC_TOOLCHAIN=$PULP_TOOLCHAIN_FOLDER
source /$PULP_SDK_FOLDER/sourceme.sh
source /$PULP_SDK_FOLDER/configs/gap.sh
source /$PULP_SDK_FOLDER/configs/platform-xxx.sh
```

Run the *Makefile* of the desired method to execute inference with `make clean all run` inside the chosen folder.

To modify the execution with new features or initialize with your pretrained model refer to the corresponding *cluster.c* file of the selected method.

## dense

Implements regression using HD dense binary vectors.
Refer to *cluster.c* for features declaration, main functions execuing projection and inference on pretrained model.

## mlp

Infereence using a pretrained Multi-Layer Perceptron.
Refer to *cluster.c* features definition and inference.

## rafe

Implements regression using HD sparse binary vectors.
Refer to *cluster.c* for features declaration, main functions execuing projection with RAFE algorithm and inference on pretrained model.

## skc

Implements regression using HD sparse binary vectors.
Refer to *cluster.c* for features declaration, main functions execuing projection with SKC algorithm and inference on pretrained model.

## regression

Implements regression using quadratic regression algorithm.
Refer to *cluster.c* for features declaration, model initialization and inference.
