# CHANGELOG



## v0.1.0 (2025-05-21)

### Chore

* chore: use escnn-atompaint fork ([`d68840d`](https://github.com/kalekundert/atompaint/commit/d68840d0136dc374d21c82249a0b2000d39c7407))

* chore: remove special-case installation for lie_learn ([`2a29f41`](https://github.com/kalekundert/atompaint/commit/2a29f418e4b204073ecbaa6104a4a02bb816c72a))

* chore: update release scripts ([`0979d60`](https://github.com/kalekundert/atompaint/commit/0979d604760fbfaaccdede280b73045ac9c86c31))

* chore: remove unused pymol files ([`e97d4cc`](https://github.com/kalekundert/atompaint/commit/e97d4ccf8a7dd001a0c5d865e4015dc25a9697c0))

* chore: remove unused manifest file ([`ec8aad5`](https://github.com/kalekundert/atompaint/commit/ec8aad5e237ae3701eb5e721395598e9d02352ea))

* chore: fix formatting ([`3a3efbf`](https://github.com/kalekundert/atompaint/commit/3a3efbf2cd279b1c54d43b72b71df4556667e1bf))

* chore: configure pre-commit linting ([`79ebc80`](https://github.com/kalekundert/atompaint/commit/79ebc8014fa930457e009fe96f1c41befa78694b))

* chore: update github actions ([`06745e9`](https://github.com/kalekundert/atompaint/commit/06745e9ffbf6dc117f6103f86b4a8299ced1d433))

* chore: build using flit, not setuptools

The project is now pure-python again, so no need to use setuptools. ([`8fea962`](https://github.com/kalekundert/atompaint/commit/8fea9626772a85b54091a177886fd57cff4d4b1b))

* chore: require C++14 instead of C++20

I got rid of the &lt;format&gt; header in the previous commit, and that turned
out to be the only feature that required the C++20 standard. ([`a1cc5c7`](https://github.com/kalekundert/atompaint/commit/a1cc5c779c3feb0bf728e7d12479f0ede90ed00f))

* chore: remove debug() statements from tests ([`a8b72ce`](https://github.com/kalekundert/atompaint/commit/a8b72cee43f44459b25ffa382a1e17622dcbc839))

* chore: add Eigen 3.4.0 as a submodule ([`fb7a9ba`](https://github.com/kalekundert/atompaint/commit/fb7a9ba22f98b104421ca50de4e011240a919698))

* chore: add pyarrow dependency ([`3512604`](https://github.com/kalekundert/atompaint/commit/3512604aa15f3026f9a97e9d8e31cdcb01d8a008))

* chore: add tensorboard dependency ([`e902658`](https://github.com/kalekundert/atompaint/commit/e902658d989a329aa7f938c88de4532bad4153ce))

* chore: bump escnn dependency ([`fd26207`](https://github.com/kalekundert/atompaint/commit/fd26207edc83dc0a914dba80e80a35083e815d2f))

* chore: add matplotlib as a test dependency ([`90b3914`](https://github.com/kalekundert/atompaint/commit/90b3914fd0066be21d60709cc182645e7f7380e8))

* chore: update gitlint rules ([`7e21e8b`](https://github.com/kalekundert/atompaint/commit/7e21e8baf120bd7f86987685383ac6242d58eae9))

* chore: switch to gitlint ([`215b1ec`](https://github.com/kalekundert/atompaint/commit/215b1ece7b0978a12ebe4224ddc6f0b119b73424))

* chore: fix CI testing pipeline ([`f146295`](https://github.com/kalekundert/atompaint/commit/f146295d681b9e6bf4d4f7e3eba369017cc6093c))

* chore: only test supported versions of python ([`5f2b00b`](https://github.com/kalekundert/atompaint/commit/5f2b00bfa2ed708da4b2a59d5a4d6b56d4974f8a))

* chore: install lie_learn from GitHub ([`f035826`](https://github.com/kalekundert/atompaint/commit/f03582626cc3a6e2aff820c798ec4c691ef23cb7))

* chore: apply cookiecutter ([`c5a43ea`](https://github.com/kalekundert/atompaint/commit/c5a43ea90cab3a9ebb8973e73b06af4061a18c54))

### Feature

* feat: implement sequence recovery metric ([`ad5509c`](https://github.com/kalekundert/atompaint/commit/ad5509cefb6ae28a5b2911535b7618212447874f))

* feat: implement an end-to-end training loop ([`6c88067`](https://github.com/kalekundert/atompaint/commit/6c880675a5db77938a0cc3c50a7d83141820cec1))

* feat: update `make_sample()` signature ([`016fbe7`](https://github.com/kalekundert/atompaint/commit/016fbe7ff730c7f5cd6af0d7a1323f8a34950c36))

* feat: make cropped amino acid images ([`859f9d0`](https://github.com/kalekundert/atompaint/commit/859f9d0bd03bd4a2c2ac1230beb85f559910f789))

* feat: log parameter magnitudes ([`6cc5417`](https://github.com/kalekundert/atompaint/commit/6cc54173068f9d7b41ac17400ed2c6877ede31c5))

* feat: make UNets where channels are set by up/downsample blocks ([`41a351a`](https://github.com/kalekundert/atompaint/commit/41a351a76b042f3c3a659d1218ca91d23872f1c6))

* feat: make amino acid sample factories more customizeable ([`fb1729a`](https://github.com/kalekundert/atompaint/commit/fb1729addd97b3916f035cf29019326a0ca8b9ad))

* feat: make the amino acid classification dataset ([`164acd0`](https://github.com/kalekundert/atompaint/commit/164acd0c2929baaf1650226a99d0d1979f566de8))

* feat: more easily control mid-type of gamma blocks ([`1b941cb`](https://github.com/kalekundert/atompaint/commit/1b941cbe5a640ac0af7f81eef10b1fc672a034ec))

* feat: add `flatten_base_space()` to public API ([`9fbed19`](https://github.com/kalekundert/atompaint/commit/9fbed1985cb735668be1284980c02f391b1cf521))

* feat: allow N-dimensional sinusoidal embeddings ([`57b459e`](https://github.com/kalekundert/atompaint/commit/57b459e0c584db2377bab615241e189dc6251114))

* feat: use optimized diffusion parameters (expt 120) ([`94a794e`](https://github.com/kalekundert/atompaint/commit/94a794e935590ae5494e0b1282939270f13cc8f8))

* feat: don&#39;t assign devices when loading checkpoints

Whoever instantiates the model should be responsible for moving it to
the proper device. ([`8d1bc4f`](https://github.com/kalekundert/atompaint/commit/8d1bc4fa54ada737c6b637813ba42841f1b287b9))

* feat: add pretrained models from expt 102 ([`155a723`](https://github.com/kalekundert/atompaint/commit/155a723d54d261c0d8c9420a92f11814a47ee862))

* feat: implement inpainting with resampling steps

This is the inpainting algorithm described by [Lugmayr2022], modified to
work with [Karras2022].  See experiment 116 for validation data.

This commit also fixes two bugs:

- Previously, the previous *noisy* images was used as the
  self-conditioning input during generation.  Now, the previous *clean*
  image is used.

- The random labels used during validation were heavily biased towards
  DNA and RNA.  Now each possible label is equally weighted. ([`1d37453`](https://github.com/kalekundert/atompaint/commit/1d37453cfc010925fecb0bf9f216b7db5f5a5656))

* feat: add support for learning rate schedules ([`bd96d8d`](https://github.com/kalekundert/atompaint/commit/bd96d8d67d325f5e714645bce419b25fc4c6191b))

* feat: implement `make_sample()` function for VAEs ([`de487ed`](https://github.com/kalekundert/atompaint/commit/de487edb461af3a03deeebde7108bf91d9bdfcbc))

* feat: condition on labels and prior diffusion steps ([`83dd209`](https://github.com/kalekundert/atompaint/commit/83dd209ef96309b4bea9214d5b5a7e6760c83398))

* feat: add option to skip generative metrics ([`10f8cf3`](https://github.com/kalekundert/atompaint/commit/10f8cf35882cfa741cd4d96baaf2ac53a030906b))

* feat: send generative metrics progress bar to stdout ([`fc3cace`](https://github.com/kalekundert/atompaint/commit/fc3cace52ad36856d0efef55410a97efb66e9be5))

* feat: change how VAE MSE loss term is reduced

See Experiment #104 for details. ([`92a592b`](https://github.com/kalekundert/atompaint/commit/92a592b7181a388f55be790964115c2fe61973f0))

* feat: add variational autoencoders ([`d6f9e2c`](https://github.com/kalekundert/atompaint/commit/d6f9e2cf621b9380357b1464ee8404340c960008))

* feat: generalize the encoder API ([`0a371b4`](https://github.com/kalekundert/atompaint/commit/0a371b474c64f57a70a93324a7951a89f3b0e41b))

* feat: publish path to &#34;official&#34; artifact directory ([`6ca86ea`](https://github.com/kalekundert/atompaint/commit/6ca86ea76c1d5c7a08c898bbe3ee76cc83af9592))

* feat: run neighbor loc metrics during Karras training ([`fe84b2c`](https://github.com/kalekundert/atompaint/commit/fe84b2ca85f8a2cc5f09f783630287a15b052714))

* feat: specify U-Net blocks using lists-of-lists ([`ff269dc`](https://github.com/kalekundert/atompaint/commit/ff269dc20aa7b25a68f5710edb8367a46730f245))

* feat: add a Fréchet distance metric ([`58c9d29`](https://github.com/kalekundert/atompaint/commit/58c9d291a1d148c90a5ec5d0a65e4b2ce3b11c92))

* feat: add the neighbor location accuracy metric ([`7ca4884`](https://github.com/kalekundert/atompaint/commit/7ca48848d99091304270f606be323d2ee259f171))

* feat: add some convenience layers ([`18d6a28`](https://github.com/kalekundert/atompaint/commit/18d6a285bf37830fd268bf9bf17da4a94d82af4e))

* feat: reorganize the neighbor location classifiers ([`1101846`](https://github.com/kalekundert/atompaint/commit/1101846735a910a641f06be53918cb48a3fbb207))

* feat: implement equivariant/non-equivariant U-Nets ([`183d127`](https://github.com/kalekundert/atompaint/commit/183d12749a1905daf9b16d2d2fa3c96b0a326bcf))

* feat: implement [Ho2020] and [Karras2022] diffusion models ([`2670d1d`](https://github.com/kalekundert/atompaint/commit/2670d1d5ca4ef7818ce1b28cd9e612ae13d7e60f))

* feat: don&#39;t hard-code the data variance ([`7549f18`](https://github.com/kalekundert/atompaint/commit/7549f18f5c28f39c22a0542763ddfa07a2061e60))

* feat: implement an equivariant U-Net ([`d4b599f`](https://github.com/kalekundert/atompaint/commit/d4b599f8be0c87c87d6fbfa739310a2a9428dca4))

* feat: implement an initial diffusion model ([`31c1cdd`](https://github.com/kalekundert/atompaint/commit/31c1cddab2a1ced2713fa44fc55031c4eecba6b4))

* feat: remove code that was spun off into a standalone library ([`8af3e44`](https://github.com/kalekundert/atompaint/commit/8af3e44e3d75b30b287358823e9dd2b41464a780))

* feat: implement non-equivariant classifier for CNNs ([`0449f2b`](https://github.com/kalekundert/atompaint/commit/0449f2b4bb6f87c36e3d173c5d1dfb3eeecb3c3f))

* feat: log the number of data loader processes ([`fd71421`](https://github.com/kalekundert/atompaint/commit/fd71421c136489348dadda2a8f2e0b7272dda384))

* feat: facilitate fine-tuning experiments ([`62978fe`](https://github.com/kalekundert/atompaint/commit/62978fe31193e2948f66c0d4188d894ac65cd852))

* feat: implement WRN-inspired beta ResNets ([`23aaa9e`](https://github.com/kalekundert/atompaint/commit/23aaa9ea519bc6cab07be62fd0d1d119898e870a))

* feat: add exact width field types ([`9dd841d`](https://github.com/kalekundert/atompaint/commit/9dd841d72f9edee5dfa187f5dcce966737f6707e))

* feat: use persistent worker subprocesses

Lightning recommends enabling the perseitent worker setting when using
&#34;spawn&#34;-style subprocesses, because these subprocesses take longer to
start and this setting reduces the number of times that needs to happen.

I enabled this setting partially to follow Lightning&#39;s recommendation
(i.e. to silence a warning message) and also partially because I was
experiencing deadlocks.  This temporarily seemed to avoid the deadlock,
but I really doubt it addressed the underlying problem, so I expect I&#39;ll
still have to deal with this eventually. ([`3ef41a0`](https://github.com/kalekundert/atompaint/commit/3ef41a00c0d98a02b6457d47d14b829f1eae2fed))

* feat: implement non-equivariant classifier for ResNets ([`139d534`](https://github.com/kalekundert/atompaint/commit/139d534055add294198b5c5c837d2dd39052f40f))

* feat: implement a non-equivariant CNN ([`f27920e`](https://github.com/kalekundert/atompaint/commit/f27920ea6dce9f854ab84b496c8eeed24a1b64fd))

* feat: don&#39;t plot if hparam value is null ([`da44c0f`](https://github.com/kalekundert/atompaint/commit/da44c0f5533befb27ae1475236a16cc658af3091))

* feat: add a non-equivariant classifier ([`99ec187`](https://github.com/kalekundert/atompaint/commit/99ec187bb888ee20e5710cdbf576772a7c2c7a7b))

* feat: test different nonlinearities in the DenseNet architecure ([`83d2126`](https://github.com/kalekundert/atompaint/commit/83d2126f31201ae15555201587635d6b8270b18b))

* feat: add a script for ploting training metrics ([`3ee6add`](https://github.com/kalekundert/atompaint/commit/3ee6add6d3d214acde29c052e2efc485c6c69f33))

* feat: manage random seed during equivariance tests ([`6b18cde`](https://github.com/kalekundert/atompaint/commit/6b18cdea25b7d0a01683b722f190e4ec7b8907d5))

* feat: support equivariance testing for low dimensional outputs ([`bcfae31`](https://github.com/kalekundert/atompaint/commit/bcfae319d5f036c96d99385f888b9432c3539933))

* feat: add a trivial field type factory ([`c0ea7f8`](https://github.com/kalekundert/atompaint/commit/c0ea7f850c23f8f52864c90027deec331cbca082))

* feat: allow different pooling factors after each DenseBlock ([`01981ee`](https://github.com/kalekundert/atompaint/commit/01981eec0757060130fc63636ddfd43a3b55101f))

* feat: implement the DenseNet architecture ([`b97ad78`](https://github.com/kalekundert/atompaint/commit/b97ad788e7485d5004ebf0cbaa21103f391b9894))

* feat: add leaky hard-shrink nonlinearity ([`fe76889`](https://github.com/kalekundert/atompaint/commit/fe76889bc5410eb0a3a4f91682400537d288bb2f))

* feat: implement the ResNet architecture ([`262790f`](https://github.com/kalekundert/atompaint/commit/262790f6d1b2a651ac01bffbfdb1e03234d6fac5))

* feat: submit multiple jobs at once ([`0109324`](https://github.com/kalekundert/atompaint/commit/01093240b3715b0c751b52c90eafaddfb9ebfb9d))

* feat: control output directory via environment variable ([`f1b03d6`](https://github.com/kalekundert/atompaint/commit/f1b03d6d0a4bd28b7e6f6a6ed351f3ee1a882231))

* feat: requeue once there&#39;s not enough time for another epoch ([`744ba25`](https://github.com/kalekundert/atompaint/commit/744ba25cd77bf8496bc8fbc44d72e77c69f99f25))

* feat: adda sbatch helper script ([`3a65d7e`](https://github.com/kalekundert/atompaint/commit/3a65d7e4f25bb5bebe62295f7f96c0e948b90b41))

* feat: add a convolution padding hyperparameter ([`a78181c`](https://github.com/kalekundert/atompaint/commit/a78181cbf7fd4f2e8c5bc5c4feac4130a22d09c7))

* feat: improve API for accessing manual predictions ([`7da6250`](https://github.com/kalekundert/atompaint/commit/7da6250d5b73a00edd7e54917dbe01d213ef13ea))

* feat: log some rare but important events ([`d8e0708`](https://github.com/kalekundert/atompaint/commit/d8e070883e98a665701ccaf6a69d2018bb6c4c99))

* feat: resume training from the last completed epoch

- Implement a dataset that can yield an infinite number of training
  samples, grouped into &#34;epochs&#34; of arbitrary length.

- Implement an epoch-aware sampler, such that different training
  examples are used in each epoch.

- Save checkpoints after each epoch.

- Account for the fact that a number of ESCNN layers must be in &#34;eval
  mode&#34; in order to successfully save and restore checkpoints.

- Properly configure logging. ([`8e772be`](https://github.com/kalekundert/atompaint/commit/8e772be698923ed652ac2b11cb6ab8f43071a197))

* feat: tweak pymol plugin UI ([`e05453b`](https://github.com/kalekundert/atompaint/commit/e05453b4d42b2f3da9a36fdb86e0430393e2151e))

* feat: add pymol plugin for manually validating training examples ([`cac03ff`](https://github.com/kalekundert/atompaint/commit/cac03ff8e760ab90054e3ad56f0eafa3362f3f4e))

* feat: add pymol plugin to visualize training examples ([`e440948`](https://github.com/kalekundert/atompaint/commit/e440948542be6b1da1f86afc43447a2585e25476))

* feat: add an option to record training examples ([`17adef2`](https://github.com/kalekundert/atompaint/commit/17adef2c787c77752b5ff8932c44eafb5eba9a80))

* feat: log classification accuracy ([`f188b76`](https://github.com/kalekundert/atompaint/commit/f188b76733154736b68006cad58c48854b678679))

* feat: switch to a classification model

- Remove the coordinate frame regression model.

- Use an inverse Fourier transform to make the classification
  predictions equivariant. ([`885c6b4`](https://github.com/kalekundert/atompaint/commit/885c6b4929f185637a547a2547da0731e890f4a3))

* feat: copy origins to /tmp before training ([`82ecda4`](https://github.com/kalekundert/atompaint/commit/82ecda489b205db0f973d5c2dd878b97bb259ef8))

* feat: reduce default polling interval to 1s ([`0981125`](https://github.com/kalekundert/atompaint/commit/098112575e7168b8ba5b0e9c23b9be4f8bbbef3c))

* feat: load training examples from an SQLite database ([`09f544f`](https://github.com/kalekundert/atompaint/commit/09f544fa27f2f4963ea4386d3593f0d21517cdb7))

* feat: make origin tags categorical to reduce memory usage ([`ba3b4e4`](https://github.com/kalekundert/atompaint/commit/ba3b4e4e4e71450f77ceca54028dd66d26793d69))

* feat: add a flexible CLI for plotting memory profiling data ([`0ad5230`](https://github.com/kalekundert/atompaint/commit/0ad5230006d83e7099725c0eb4ce81551a354bc4))

* feat: add memory mapping profiler ([`84ba24e`](https://github.com/kalekundert/atompaint/commit/84ba24ea4d2f20f27b11ec3efe9714c25e096cdd))

* feat: add distinct training, validation, and test datasets ([`9a9e79d`](https://github.com/kalekundert/atompaint/commit/9a9e79d998173aa77192ed3fc3d9fcabae08015f))

* feat: add training script ([`0bf26ce`](https://github.com/kalekundert/atompaint/commit/0bf26ce06a1feb7810ff5087c377eba1f9c5233e))

* feat: exclude nonbiological ligands when choosing origins ([`3151b5e`](https://github.com/kalekundert/atompaint/commit/3151b5eb7404a196347da15489857a6131cf22e6))

* feat: indicate how many structures are missing ([`679eddd`](https://github.com/kalekundert/atompaint/commit/679eddd02e86092c9d54a7ef08c0e6ebc146a0d2))

* feat: make ap_choose_origin easier to run in parallel ([`a5a544d`](https://github.com/kalekundert/atompaint/commit/a5a544dbb2e403fbe3e5494a839f0a523e6937f5))

* feat: implement neighbor count data stream ([`4d48604`](https://github.com/kalekundert/atompaint/commit/4d48604f77b72828563db2bbdaa014bf7b7e86d5))

* feat: implement equivariant CNN models ([`9d5ee76`](https://github.com/kalekundert/atompaint/commit/9d5ee76e24e7117ef72b80fa105e3172b83da54e))

### Fix

* fix: include image in amino acid coordinate sample ([`a428e15`](https://github.com/kalekundert/atompaint/commit/a428e150df3a8809423c474f23d5e7fabd8f91be))

* fix: use correct dtypes for amino acid samples ([`600c087`](https://github.com/kalekundert/atompaint/commit/600c087961d7adc4006d0fc42ef601bf27e9cf58))

* fix: filter out residues with multiple amino acid types ([`43477d5`](https://github.com/kalekundert/atompaint/commit/43477d5a485bdd4be3d0d1d4f90d7ae8dad77320))

* fix: use new API for conditioning ([`15a8a2d`](https://github.com/kalekundert/atompaint/commit/15a8a2d5cf0590de860cbee606e0f3c2ae89243e))

* fix: use new API when generating images during training ([`ad4039c`](https://github.com/kalekundert/atompaint/commit/ad4039c48bb8fec626b569ce543f18a7445f7dda))

* fix: move model to given device when loading weights ([`7783fd9`](https://github.com/kalekundert/atompaint/commit/7783fd92a6f5172111e3f1b9f3e6492eb5d82f8b))

* fix: invert the meaning of the mask voxels

Previously, the inpainting algorithm would use the known image where the
mask was 1, and generate new voxels where the mask was 0.  However, this
was less-than-ideal for two reasons:

1. It&#39;s the opposite of the usual meaning of the word &#34;mask&#34;.  The mask
   should cover be what the inpainting algorithm *can&#39;t* see, not what
   it *can*.

2. It&#39;s easier to visualize masks in pymol when only the to-be-hidden
   parts are non-zero. ([`8cf629a`](https://github.com/kalekundert/atompaint/commit/8cf629a8ce9833c11e78148447c0fc08fabbcf4f))

* fix: sample polymer/CATH labels with uniform weights ([`74e725a`](https://github.com/kalekundert/atompaint/commit/74e725ae6a068ede35fc75784df2829f0ebe1bc6))

* fix: debug image generation with self/label conditioning ([`c34bf87`](https://github.com/kalekundert/atompaint/commit/c34bf87ce590dc109a30159dd79424ba07666614))

* fix: include CATH label in randomly sampled labels ([`b204db6`](https://github.com/kalekundert/atompaint/commit/b204db6996b63dab36130ead7bd0b35e9f079746))

* fix: debug self-conditioning for each UNet ([`b4cdb2f`](https://github.com/kalekundert/atompaint/commit/b4cdb2fc5a3907ef5969818ed914f3429f0a86a7))

* fix: adjust semisym UNet input size for self conditioning ([`f116bbf`](https://github.com/kalekundert/atompaint/commit/f116bbfa5e727b85a33927af7e7917dbd0e3a94c))

* fix: use `np.concatenate()` instead of `np.concat()`

The former is compatible with numpy&lt;2.0, while the latter is not. ([`cb97032`](https://github.com/kalekundert/atompaint/commit/cb9703248a6d219880e76a6782719562bddbb040))

* fix: update import path ([`5ce11c9`](https://github.com/kalekundert/atompaint/commit/5ce11c9d2fea1fbfea92aca57c85e79d2e7ef387))

* fix: specify `weights_only` argument to `torch.load()`

The default will be changed in a future version. ([`6a9b4fb`](https://github.com/kalekundert/atompaint/commit/6a9b4fbf4d413794ca1d557fe5fe13d3ac965c36))

* fix: sync metrics each epoch ([`7122224`](https://github.com/kalekundert/atompaint/commit/712222460dac2f4a68d13f84242558c60244fb14))

* fix: make the Fréchet distance metric compatible with DDP ([`a8d4767`](https://github.com/kalekundert/atompaint/commit/a8d47677b4170fc656790cfac243fe1e4ae30cb6))

* fix: don&#39;t allow non-finite sigma values

Prior to this commit, I was using a bespoke method of generating samples
from a Gaussian distribution.  On two separate occasions, this method
was responsible for weird bugs stemming from its ability to generate
non-finite values.  While I think I&#39;ve now encountered every non-finite
value my method could generate, I decided that it was a mistake to use a
bespoke method in the first place, so I switched to the NumPy
implementation.  This requires some machinery to conveniently batch
NumPy `Generator` objects, which is why I didn&#39;t take this approach in
the first place, but it&#39;s not too bad. ([`bae3c71`](https://github.com/kalekundert/atompaint/commit/bae3c71dccbec1cf464bc74682172f836a001c48))

* fix: avoid segfaults when calculating Fréchet distance ([`ff66b5f`](https://github.com/kalekundert/atompaint/commit/ff66b5f5bd4769c4ca52c3d0341b9663a3d735e7))

* fix: minor issues for running on HMS O2 ([`7353c7c`](https://github.com/kalekundert/atompaint/commit/7353c7c99d06163f6f6618abb7a7b5476a6e59e0))

* fix: add missing dependency ([`3b1747b`](https://github.com/kalekundert/atompaint/commit/3b1747b50011d02db9051c2249315a3fa8f31cab))

* fix: avoid `x[:, *y]` syntax not supported in python 3.10 ([`5061948`](https://github.com/kalekundert/atompaint/commit/5061948dbcd9342015fa47207fcff4bc8b9aabcc))

* fix: ensure uniform random values are in [0, 1)

Previously I was generating 64-bit random values in this range, then
downcasted them to 32-bits.  This inadvertently broadened the range to
[0, 1].  Now I&#39;m directly generating 32-bit values in the correct range. ([`b84ae8b`](https://github.com/kalekundert/atompaint/commit/b84ae8b3bf0eaaf71a8971718c48e9f04396e1ef))

* fix: incorporate new escnn bug fixes ([`01ef0a4`](https://github.com/kalekundert/atompaint/commit/01ef0a4ab439fba3dc0186df130347bf498c02db))

* fix: wrong variable name ([`7bd4236`](https://github.com/kalekundert/atompaint/commit/7bd4236b5726528675f97424e5439215088867fd))

* fix: avoid tensor product of zeroth and first irrep ([`53c14fa`](https://github.com/kalekundert/atompaint/commit/53c14fa1fe006703b931626ef492793f8b9df6dd))

* fix: avoid using the &lt;format&gt; header

This header is not present in the GCC standard library until GCC&gt;=13
[1].  The latest version of GCC available on the cluster is 9.2.

[1] https://en.cppreference.com/w/cpp/20 ([`3a4f36d`](https://github.com/kalekundert/atompaint/commit/3a4f36d1c01ac4f68d2b4d09468fcfc15580e0d1))

* fix: only set mutliprocessing context if multiple workers ([`4c6b8a5`](https://github.com/kalekundert/atompaint/commit/4c6b8a5ab9ff6a9d4b038ff33c6dc153f3a8b10d))

* fix: spawn (rather than fork) dataloader worker processes

This seems to fix the race conditions/deadlocks/abort signals that have
been causing problems for the last few days.  I still don&#39;t know exactly
what the issue is, but spawning brand new processes is generally much
safer than forking the current process, and the memory/startup time
penalties aren&#39;t significant in the context of such a long-running
program.

In order to make this change, I had to move the code for copying the
origins database to /tmp from inside the worker processes to outside. ([`d105f7e`](https://github.com/kalekundert/atompaint/commit/d105f7e7e3b1819793b44ec3aa2a441d8def69b9))

* fix: avoid SIGABRT in the context of multiprocess data loading

See comment in code for more details about this weird, and
not-really-fixed, bug. ([`395f567`](https://github.com/kalekundert/atompaint/commit/395f5673d54ffe1be0cfe8291456802d858e3431))

* fix: only use complete minibatches ([`d1e548b`](https://github.com/kalekundert/atompaint/commit/d1e548b8b26b8697f29bf7e774fd14b497fbfc62))

* fix: update type hint ([`9ddfb9a`](https://github.com/kalekundert/atompaint/commit/9ddfb9aa68b2d86cd36d531391b8198458a4bc2b))

* fix: off-by-one error when selecting origins ([`9509537`](https://github.com/kalekundert/atompaint/commit/950953712eadfbc1135836f0a04d9afa88168eae))

* fix: don&#39;t exceed `max_tries` ([`c39b76f`](https://github.com/kalekundert/atompaint/commit/c39b76f550af1148612fbd4808b0fc192fe53534))

* fix: make test epoch the right size ([`ff5b9ec`](https://github.com/kalekundert/atompaint/commit/ff5b9ec540c19c09281d1a70b812e80f5024807b))

* fix: initialize InfiniteSampler with a valid epoch

I originally avoided doing this, so I&#39;d get an error if the sampler was
used before `set_epoch()` was called.  Unfortunately, it turns out that
Lightning does draw examples from the data loader before setting the
epoch.  This occurs when Lightning is &#34;checking that the data loader is
iterable&#34;.  I don&#39;t think the resulting training examples are actually
used for anything, though.

For reasons I don&#39;t understand, this is only a problem when using
multiple data loader processes.  With only a single data loader,
Lightning still appears to do this check, but apparently doesn&#39;t require
the sampler to be invoked somehow. ([`cd06068`](https://github.com/kalekundert/atompaint/commit/cd060689823d39f83f9901b452834d5ccb38fc6f))

* fix: update out-of-date import statement ([`297e65a`](https://github.com/kalekundert/atompaint/commit/297e65a7e3788ca23715787b2790391323e35712))

* fix: restore API used by visualization scripts ([`308ba3a`](https://github.com/kalekundert/atompaint/commit/308ba3ac38709581ac1c22fc8dea23dc12da08f9))

* fix: correctly import exception object ([`3112366`](https://github.com/kalekundert/atompaint/commit/31123666cf96ac20a86e5a3956d04ec6b92d1144))

* fix: refactoring bugs in the memory profiler ([`eb4ffd5`](https://github.com/kalekundert/atompaint/commit/eb4ffd5db31365639dbcb1dd3c07d4c84b8cf06e))

* fix: don&#39;t import expensive packages at top level ([`6f2fc18`](https://github.com/kalekundert/atompaint/commit/6f2fc1819c26c018b0cd768cd63d5cf1169ce44f))

* fix: check that a process exists before profiling its memory ([`9c8ba37`](https://github.com/kalekundert/atompaint/commit/9c8ba376f5bd1471a3a3fa079c91122d1b345133))

* fix: update path to ap_choose_origins entry point ([`eab950c`](https://github.com/kalekundert/atompaint/commit/eab950cff0fb5b2b50537e1c222f9bd6834c13a1))

* fix: use multiple processes to load training examples ([`928b555`](https://github.com/kalekundert/atompaint/commit/928b555db074a218b05e2f0dcf4e945e4ca20c30))

* fix: sample origins more quickly

- Pre-calculate which origins belong to which tag.

- Assign all origins the same chance of being picked.  This means that
  the &#39;weight&#39; column in the origins data frame is now ignored.
  Eventually, I&#39;ll want to rewrite the origin-picking code to remove
  this column completely. ([`7562cc5`](https://github.com/kalekundert/atompaint/commit/7562cc579d505bbcc8766da278bf630fc5fff91c))

* fix: more quickly skip atoms that aren&#39;t part of the image ([`c059791`](https://github.com/kalekundert/atompaint/commit/c0597918841dd64ec7a412d474cdb0898ce49fb7))

* fix: create MLP output on same device as input ([`1341d81`](https://github.com/kalekundert/atompaint/commit/1341d81613f7267850e7a5b7a52d991271dccabf))

* fix: move all relevant tensors to the GPU ([`71fdabe`](https://github.com/kalekundert/atompaint/commit/71fdabe4edd95432ca989c74225addcab3cffb94))

* fix: correctly calculate the number of structures for each worker to process ([`e9e1849`](https://github.com/kalekundert/atompaint/commit/e9e184995c92911fd39cb53dcda53faa4ee28f40))

* fix: allow chains with multi-letter names ([`b096f07`](https://github.com/kalekundert/atompaint/commit/b096f0789b1be5b87c96a4064c1469a9a7ceaaca))

* fix: update test name ([`fb94a53`](https://github.com/kalekundert/atompaint/commit/fb94a5323e93d4a44f9e66935a70c6d14ac84f82))

* fix: minor bugs relating to ap_choose_origins ([`69eaa62`](https://github.com/kalekundert/atompaint/commit/69eaa6226669ecd285eb22d2f3187f761f4c1c2e))

### Performance

* perf: reimplement voxelization in C++

- Improve data loading speed, maybe.  The old numba implementation was
  pretty fast, but the new C++ implementation should be faster because
  Eigen makes it possible to avoid allocating a bunch of intermediate
  arrays, and in some cases to allocate arrays on the stack.

- Remove the `numba` dependency.  I anticipate that this could make
  installation and debugging a little easier.

- Change LICENSE to GPLv3, so that `overlap` can be included in the
  distribution.

- Use `setuptools` as the build backend. ([`d4b0e6e`](https://github.com/kalekundert/atompaint/commit/d4b0e6e4b1e41aed55e7b570d8c8abace831ced9))

* perf: use faster but lower-precision matrix multiplications ([`97bc6c4`](https://github.com/kalekundert/atompaint/commit/97bc6c4e76ce52a344987a3ed5c9a216d3be229f))

* perf: don&#39;t calculate bias before batch normalization ([`68cb0fc`](https://github.com/kalekundert/atompaint/commit/68cb0fcc3b144208009a98744e47926531f3031d))

* perf: read atom coordinates from pre-cached feather files ([`8be706d`](https://github.com/kalekundert/atompaint/commit/8be706d54018e8cf7166a2c029e1791ad7583f11))

* perf: JIT-compile functions for calculating voxel intensities ([`c222b85`](https://github.com/kalekundert/atompaint/commit/c222b85d2b66e95d492997ba31993bba6838952b))

### Refactor

* refactor: simplify stack traces ([`ff7660c`](https://github.com/kalekundert/atompaint/commit/ff7660cd140d75c851e4e0c7a5e402b9f250a86a))

* refactor: remove the pytorch3d dependency ([`a2b52c2`](https://github.com/kalekundert/atompaint/commit/a2b52c2b7c97e24277cdddae1476b891e464eebc))

### Test

* test: use zero tensor when initializing self-conditioning ([`70ad591`](https://github.com/kalekundert/atompaint/commit/70ad591d2b713e89aedc763859e62bd2bcfd7d7e))

* test: fix flaky KL divergence test

Hypothesis was sometimes able to find some obscure corner cases that
triggered failures despite not representing a significant difference. ([`796355a`](https://github.com/kalekundert/atompaint/commit/796355a77ad17832ee2e94d5e77229a234bcad76))

* test: increase tolerance for flaky test ([`3d7f8b9`](https://github.com/kalekundert/atompaint/commit/3d7f8b90d56b671f5ec70d2abb20512d56a223ac))

* test: confirm beta ResNet equivariance ([`7d90bb6`](https://github.com/kalekundert/atompaint/commit/7d90bb6bd5ae2daa00576728c71cd61792f60c91))

### Unknown

* wip: switch from regression to classification

This commit contains two strategies for picking view slots that ended up
not working.  These can be found in `_classification.py.bkup` and
`classification.py`.  I&#39;m about to delete all of this code, but I wanted
to keep it available in the project history first.

This commit also refactors the transformation-prediction dataset module,
and includes updated tests. ([`0236bec`](https://github.com/kalekundert/atompaint/commit/0236becd7a9901913b45cf65a3b59f21de92bfc0))

* Merge branch &#39;master&#39; of github.com:kalekundert/atompaint

Conflicts:
	atompaint/diagnostics.py ([`8b5debd`](https://github.com/kalekundert/atompaint/commit/8b5debdb5605772da1a15792fe34cbaaccff2022))

* Merge branch &#39;master&#39; of github.com:kalekundert/atompaint ([`3eb5319`](https://github.com/kalekundert/atompaint/commit/3eb5319f1a7243266bc5c4a971579fb4242735f5))

* Merge branch &#39;master&#39; of github.com:kalekundert/atompaint ([`e52621c`](https://github.com/kalekundert/atompaint/commit/e52621cc6863c386c7ecea527dd4bf1238a326be))

* Merge branch &#39;fix-ep&#39; ([`cc0c36d`](https://github.com/kalekundert/atompaint/commit/cc0c36d0833e9acefaa8064bf930a041ff158851))

* Merge branch &#39;master&#39; of github.com:kalekundert/atompaint ([`230def1`](https://github.com/kalekundert/atompaint/commit/230def1ed05e0756634c028a3909b9e763eec455))
