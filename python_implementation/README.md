Copyright (C) 2020 ETH Zurich, Switzerland. 
SPDX-License-Identifier: GPL-3.0-only.
See LICENSE file for details.

Author: Edoardo Mello Rella <edoardom@student.ethz.ch>

# Python Implementation

Containing Python implementations of the algorithms discussed in the paper.

The complete pipeline from reading Dynamic Vision Sensors sequences to training and testing is implemented.

# Requirements

Set-up conda environment using *utils/hd_dvs.yml* file: 

`$ conda env create -f utils/hd_dvs.yml -n my_hd_dvs_env`

Required installation of [pydvs](https://github.com/better-flow/pydvs) and [pyhdc](https://github.com/ncos/pyhdc) library.

# Dataset

The preprocessed version of MVSEC autonomous driving event-based dataset is available here:

* [outdoor_day_1](https://drive.google.com/file/d/1cLks_SqnLbqSHRwPWOJn2tuRj4pasAay/view)
* [outdoor_day_2](https://drive.google.com/file/d/1rBj6gkgCSMTXO1rWs4--AF-4jwdXxFPq/view?usp=sharing)
* [outdoor_night_1](https://drive.google.com/file/d/158ZHB1NsX3Al7_59BUPVfCIUKOb09kL3/view?usp=sharing)
* [outdoor_night_2](https://drive.google.com/file/d/1Aq5SDZmQdA3GbN6lJiiJxVOQdHVpbIRW/view?usp=sharing)
* [outdoor_night_3](https://drive.google.com/file/d/1nYHvRaLmhQkCaMo6Q7LMgDOlybIyKYII/view?usp=sharing)

You can find he complete dataset [here](https://daniilidis-group.github.io/mvsec/).

# Usage

Use **_encoder.py* files to to encode DVS sequences into HD vectors in the desired way and save them. See comments inside files for complete usage instructions.

Use **_train.py* to execute training and inference with previously saved HD vectors in te desired way. See comments inside files for complete usage instructions.

Use *online_model.py* to use online update. See comments inside files for complete usage instructions.

## Sample Usage

Example usage to execute features embedding, training and testing with sparse HD vectors RAFE projection algorithm using CDT

```
$ python sparse_encoder.py --base_dir $DATA_DIRECTORY --method rafe
$ python hdv_sparse_train.py --base_dir $DATA_DIRECTORY --data_type rafe --cdt True
```