# tiersweep-16 Kaggle provenance

Provenance for the tiersweep-16 Kaggle run: cheap-tier routes L0/L1/C0 on Qwen2.5-3B/7B over the committed real Tier-A corpus. The 1.5B tier is the committed realdata-15 run and is not re-measured; its rows predate server attestation and are flagged 'unattested' by scripts/attestation_audit.py.

- Kernel: `abhigyan1818/tiersweep16-local-sweep` (branch `fm/costsmart-tiersweep-16`, git `e282989062723296bdd14808f71ef8f06c6ea5e6`).
- Download: `kaggle kernels output abhigyan1818/tiersweep16-local-sweep -p /tmp/kout`.

## Server-attested models

| model | resolved revision | weights sha256 | licence | measured rows |
|---|---|---|---|---|
| Qwen/Qwen2.5-3B-Instruct | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | `04a84318cdec5543...` | other (qwen-research) | 2400 |
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | `175d8c141750c8e6...` | apache-2.0 | 2400 |

The 1.5B tier reuses `results/realdata-15/{sweep,repeats}.db`; those rows predate attestation and are flagged `unattested`, not falsified.

## Artifact hashes

- `sweep-3b.db`: `25f7ada857b92ac727f36d42a140e5a90aca48fd1c20212eff9c3b1c377d2297` (1150976 bytes)
- `repeats-3b.db`: `62382d346d1a771900ea20d18731f79a1db2d43e03b7507ea70fd4e6c50eb052` (1945600 bytes)
- `sweep-3b.csv`: `dab869086a00ed3f65e5d60fec906eeef836439e39af44597890e95435fc333c` (837203 bytes)
- `repeats-3b_summary.json`: `252c0b2da430de85e82d38b340de35c664867c98df05101c777ed3804b36b87f` (383 bytes)
- `sweep-7b.db`: `8d597c2ad27ed607fd2d45d3dcc4e5728644d6f3d97302690d766d36e60429fa` (1208320 bytes)
- `repeats-7b.db`: `52539ef8a7f5cb0a7bed4f8753ebdbe164bd04d47bbfc5ac5efad13abd741f6d` (2146304 bytes)
- `sweep-7b.csv`: `124c964b8f5ae4013928ab3af1f2512c1d59e239d49e6ec0c054785f63f48511` (886573 bytes)
- `repeats-7b_summary.json`: `5c40e54c6551838958552911d47ab2780388a8d65cbf176613693b04fd6d78eb` (383 bytes)
- `3b-server.log`: `c73a98b09b6ee6746d8ea9d7f8ad58d1a7dad212c861a8f902b3d45f40d21755` (124179 bytes)
- `7b-server.log`: `b4434627cf4c4dac186c9862bed4f1bc30b023527d24fc9a93a444dd193a9f43` (100437 bytes)
- `tiers.json`: `559e588225fc17f0d5a3252cc35936cb52fbaf7180fcf0ab6d2838a6e469bd27` (31589 bytes)
- `TIERS.md`: `22f71f51fdde0a1703cd6561b76c1a50866b31d08cb33063ff4ccfa9691b8b0d` (1723 bytes)
- `tiers.svg`: `b8c08d007f1f5b88df6b3c0afc870251a57f8e4f023e1dd1a67bc93f30502f7f` (2450 bytes)
- `attestation_audit.json`: `ac148b160345aa40ad0ad8b2fce222b40c449613e8f8ea06aa4f9450ddf79fcd` (11036 bytes)
- `MODEL_MANIFEST.md`: `777521c6cf767b571483731c9abeb92d8e609851091711d608b5608fb8f6bdea` (2161 bytes)

The Kaggle kernel log is not committed; re-download and verify its recorded sha256 above.
