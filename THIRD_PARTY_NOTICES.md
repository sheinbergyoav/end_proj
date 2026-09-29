# Third-Party Notices

## llama2.c
- Author: Andrej Karpathy
- Source: https://github.com/karpathy/llama2.c
- License: MIT
- Used in: `src/main.c` (inference engine structure, sampler, tokenizer handling)
  and `model/export.py` (Q8_0 export, "ak42" binary layout). Both were modified
  for this project.

## stories15M model
- Author: Andrej Karpathy (trained on the TinyStories dataset)
- Source: https://huggingface.co/karpathy/tinyllamas
- License: see the model page. Weights are NOT distributed in this repository.

## TinyStories dataset
- Authors: Ronen Eldan and Yuanzhi Li (Microsoft Research)
- Used to train stories15M. Not distributed here. See the dataset page for terms.

## Tokenizer (tokenizer.bin / tokenizer.model)
- Derived from the Llama 2 SentencePiece tokenizer (Meta Platforms, Inc.)
- Governed by the Llama 2 Community License. Not distributed in this repository.

## TurboQuant / QJL (algorithms)
- TurboQuant, and the QJL (Quantized Johnson-Lindenstrauss) technique it builds on,
  are described in published research papers. This repository is an independent
  hardware/software implementation. Please cite the original papers (see README).

## Cadence Tensilica / Xtensa
- "Cadence", "Tensilica", and "Xtensa" are trademarks of Cadence Design Systems, Inc.
- The `tie_dev2` core configuration, `.tdk` files, Xtensa Xplorer, and the xt-* toolchain
  are proprietary to Cadence and are NOT included in this repository.
  Use of the toolchain requires your own Cadence license.
- The `.tie` files in `tie/` are original work by the authors, covered by the MIT License.
