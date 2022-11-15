```shell
Traceback (most recent call last):
  File "main.py", line 171, in <module>
    inst._run()
  File "/mnt/dd/instructor/real_data/evogan_instructor.py", line 88, in _run
    self.pretrain_generator(cfg.MLE_train_epoch)
  File "/mnt/dd/instructor/real_data/evogan_instructor.py", line 133, in pretrain_generator
    pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)
  File "/mnt/dd/instructor/real_data/instructor.py", line 105, in train_gen_epoch
    self.optimize(optimizer, loss, model)
  File "/mnt/dd/instructor/real_data/instructor.py", line 190, in optimize
    loss.backward(retain_graph=retain_graph)
  File "/root/miniconda3/envs/myconda/lib/python3.7/site-packages/torch/tensor.py", line 245, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "/root/miniconda3/envs/myconda/lib/python3.7/site-packages/torch/autograd/__init__.py", line 147, in backward
    allow_unreachable=True, accumulate_grad=True)  # allow_unreachable flag
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemmStridedBatched( handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches)`

```

```shell
Starting Adversarial Training...
  0%|                                                                                                                                                                                                 | 0/3000 [00:00<?, ?it/s/mnt/dd/models/relational_rnn_general.py:174: UserWarning: Output 0 of SplitWithSizesBackward is a view and is being modified inplace. This view is an output of a function that returns multiple views. Inplace operators on such views are being deprecated and will be forbidden starting from version 1.8. Consider using `unsafe_` version of the function that produced this view or don't modify this view inplace. (Triggered internally at  /pytorch/torch/csrc/autograd/variable.cpp:547.)
  q *= (self.key_size ** -0.5)
  0%|                                                                                                                                                                                                 | 0/3000 [00:01<?, ?it/s]
Traceback (most recent call last):
  File "main.py", line 171, in <module>
    inst._run()
  File "/mnt/dd/instructor/real_data/cegan_instructor.py", line 286, in _run
    score, fit_score, select_mu = self.evolve_generator_with_temp(adv_epoch, cfg.ADV_g_step)
  File "/mnt/dd/instructor/real_data/cegan_instructor.py", line 505, in evolve_generator_with_temp
    self.variation(evo_g_step, criterionG)
  File "/mnt/dd/instructor/real_data/cegan_instructor.py", line 628, in variation
    gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
  File "/mnt/dd/models/CeGAN_G.py", line 110, in sample
    pred, hidden, next_token, _, _ = self.step(inp, hidden)
  File "/mnt/dd/models/CeGAN_G.py", line 76, in step
    gumbel_t = self.add_gumbel(self.lstm2out(out.squeeze(1)))
  File "/root/miniconda3/envs/myconda/lib/python3.7/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/miniconda3/envs/myconda/lib/python3.7/site-packages/torch/nn/modules/linear.py", line 94, in forward
    return F.linear(input, self.weight, self.bias)
  File "/root/miniconda3/envs/myconda/lib/python3.7/site-packages/torch/nn/functional.py", line 1753, in linear
    return torch._C._nn.linear(input, weight, bias)
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```



