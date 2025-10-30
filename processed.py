 p.NB_EPOCH  = 300
# p.LOSS_NICKNAME = 'MSE+SSIM'
for id_epoch in range(starting_epoch, p.NB_EPOCH):
# current_train_loss, current_train_mse, current_train_psnr, current_train_ssim = trn(p, network, optimizer, scheduler, loader_trn, nb_batch_trn, id_epoch, device, print_epoch_dir, writer)
trn(p, network, optimizer, scheduler, loader_trn, nb_batch_trn, id_epoch, device, print_epoch_dir, writer)
val(p, network, optimizer, scheduler, loader_val, nb_batch_val, id_epoch, device, print_epoch_dir, writer, model_checkpoint_output_dir)
tst(p, network, loader_tst, nb_batch_tst, id_epoch, device, print_epoch_dir, writer)  
with open(os.path.join(path_to_loss_psnr_ssim,'loss_trn.txt'),'w') as current_train_los:
    current_train_los.write(str(loss_trn))
with open(os.path.join(path_to_loss_psnr_ssim,'loss_val.txt'),'w') as current_val_los:
    current_val_los.write(str(loss_val))
with open(os.path.join(path_to_loss_psnr_ssim,'loss_tst.txt'),'w') as current_tst_los:
    current_tst_los.write(str(loss_tst))
    
with open(os.path.join(path_to_loss_psnr_ssim,'psnr_val.txt'),'w') as current_val_psn:
    current_val_psn.write(str(psnr_val))
with open(os.path.join(path_to_loss_psnr_ssim,'psnr_tst.txt'),'w') as current_tst_psn:
    current_tst_psn.write(str(psnr_tst))
    
with open(os.path.join(path_to_loss_psnr_ssim,'ssim_val.txt'),'w') as current_val_ssi:
    current_val_ssi.write(str(ssim_val))
with open(os.path.join(path_to_loss_psnr_ssim,'ssim_tst.txt'),'w') as current_tst_ssi:
    current_tst_ssi.write(str(ssim_tst))
    
with open(os.path.join(path_to_loss_psnr_ssim,'psnr_valtst.txt'),'w') as psnr_valts:
    psnr_valts.write(str(psnr_valtst))
with open(os.path.join(path_to_loss_psnr_ssim,'ssim_valtst.txt'),'w') as ssim_valts:
    ssim_valts.write(str(ssim_valtst))

writer.close()

  

# ================================================================
# === Training routine ===========================================
# ================================================================
def trn(p, network, optimizer, scheduler, loader_trn, nb_batch_trn, id_epoch, device, print_epoch_dir, writer):

  global flip_trn
  global current_train_loss,current_train_mse, current_train_psnr, current_train_ssim
  network.train()
  trn_loss_accumulator = EpochLossAccumulator()
  trn_index_accumulator = EpochIndexAccumulator()
  time_epoch = time.time()

  global sos
  # sos = torch.tensor([1737,1954,2171,2388,2605], device = device).float().repeat(p.BATCH_SIZE_TRN)\
  #       .view(p.BATCH_SIZE_TRN,5).T
  # sos = torch.tensor([1540,1687,1834,1981,2128], device = device).float().repeat(p.BATCH_SIZE_TRN)\
  #         .view(p.BATCH_SIZE_TRN,5).T
  # sos = torch.tensor([1540,1593,1646,1699,1953], device = device).float().repeat(p.BATCH_SIZE_TRN)\
  #         .view(p.BATCH_SIZE_TRN,5).T
  sos = torch.tensor([1476,1512,1548,1584,1620], device = device).float().repeat(p.BATCH_SIZE_TRN)\
          .view(p.BATCH_SIZE_TRN,5).T
  for id_batch, (sinogram_batch, ima_ref_batch, file_name_batch) in enumerate(loader_trn):
    ima_ref_batch = custom_transform(
      ima_ref_batch.to(device).float(), p.TRANSFORM_TRN_VAL_TARGETS, p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)
    # Reset gradients to zero
    optimizer.zero_grad()
    # sinogram_batch = sinogram_batch
    # ima_ref_batch = ima_ref_batch
    # Forward pass with the deep neural network
    # 用深度神经网络向前传递
    # # 检查张量在哪个设备上
    # # 创建一个张量
    # tensor = torch.randn(3, 3)
    # # 查看张量所在的设备
    # print("Tensor is on device:", tensor.device)
    # sinogram = sinogram_batch.to(device).float()
    # print("Tensor is on device:", sinogram.device)
    ima_net_batch = network(sinogram_batch.to(device).float(), sos)
    # Calculate loss, propagate gradients back through the network, and optimize
    loss = custom_loss(ima_net_batch, ima_ref_batch, p.LOSS_NICKNAME)
    loss.backward()
    optimizer.step()
    mse = torch.mean((ima_net_batch - ima_ref_batch) ** 2)   
    # maxval = torch.max(ima_ref_batch) - torch.min(ima_ref_batch)
    # psnr = 20 * torch.log10(maxval) - 10 * torch.log10(mse)
    # ssim_value = ssim(ima_net_batch, ima_ref_batch, data_range=ima_net_batch.max() - ima_net_batch.min(), size_average=True)
    ssim_value = compare_ssim(np.array(ima_net_batch.cpu().detach().squeeze()), np.array(ima_ref_batch.cpu().detach().squeeze()),win_size=None, gradient=False, data_range=1, channel_axis=0, multichannel=False, gaussian_weights=False, full=False)
    psnr = compare_psnr(np.array(ima_net_batch.cpu().detach().squeeze()), np.array(ima_ref_batch.cpu().detach().squeeze()), data_range=1)
    # Keep track of the calculated loss
    # 通过将每个批次的损失值乘以批次大小来获得总损失值，并将其传递给累积器，
    # 以便跟踪整个训练过程中的损失值。这有助于在训练过程中监控损失值的变化并进行后续分析。
    batch_size = sinogram_batch.shape[0]
    trn_loss_accumulator.update_losses(batch_size, loss.item()*batch_size)
    trn_index_accumulator.update_index(batch_size, mse.item()*batch_size, psnr.item()*batch_size, ssim_value.item()*batch_size)
    # Log the current batch
    log_batch_to_console_tensorboard_harddrive(
      id_epoch, 'Trn', trn_loss_accumulator, time_epoch, id_batch, nb_batch_trn, ima_ref_batch, ima_net_batch,
      file_name_batch, writer, flip_trn, p, print_epoch_dir)
    
 

  # Log the current epoch
  
  current_train_loss = trn_loss_accumulator.get_epoch_loss()
  current_train_mse, current_train_psnr, current_train_ssim = trn_index_accumulator.get_epoch_index()
  loss_trn.append(current_train_loss)
  log_epoch_console_tensorboard(
    writer, 'Trn', '0_training_loss', id_epoch, time_epoch, current_train_loss, current_train_psnr, current_train_ssim, new_val_loss)
  # log_epoch_console_tensorboard(
  #   writer, 'Trn', '0_training_psnr', id_epoch, time_epoch, current_train_psnr, new_val_loss)
  # log_epoch_console_tensorboard(
  #   writer, 'Trn', '0_training_ssim', id_epoch, time_epoch, current_train_ssim, new_val_loss)
  # return current_train_loss, current_train_mse, current_train_psnr, current_train_ssim 
  # Learning rate evolution
  # writer.add_scalar('0_learning_rate',optimizer.param_groups[0]['lr'],id_epoch)
  # adjust_learning_rate(optimizer, id_epoch)
  # optimizer.step()
  
  scheduler.step()



# ================================================================
# === Validation routine =========================================
# ================================================================
#def val(p, network, optimizer, scheduler, loader_val, nb_batch_val, id_epoch, device, print_epoch_dir, writer, checkpoint_file_path_and_name):
def val(p, network, optimizer, scheduler, loader_val, nb_batch_val, id_epoch, device, print_epoch_dir, writer, model_checkpoint_output_dir):
  global min_val_loss
  global new_val_loss
  global max_val_psnr 
  #global new_val_psnr
  global max_val_ssim 
  #global new_val_ssim
  global flip_val

  global loss_valtst
  global psnr_valtst
  global mse_valtst
  global ssim_valtst
  global current_val_loss, current_val_mse, current_val_psnr, current_val_ssim
  #global valshortastst_loss, valshortastst_mse, valshortastst_psnr, valshortastst_ssim
  network.eval()
  val_loss_accumulator = EpochLossAccumulator()
  val_index_accumulator = EpochIndexAccumulator()
  time_epoch = time.time()

  with torch.no_grad():
    for id_batch, (sinogram_batch, ima_ref_batch, file_name_batch) in enumerate(loader_val):
      ima_ref_batch = custom_transform(
        ima_ref_batch.to(device).float(), p.TRANSFORM_TRN_VAL_TARGETS, p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)
      sos = torch.tensor([1476,1512,1548,1584,1620], device = device).float().repeat(sinogram_batch.shape[0])\
        .view(sinogram_batch.shape[0],5).T
      # Forward pass with the deep neural network
      ima_net_batch = network(sinogram_batch.to(device).float(), sos)
      # Loss
      loss = custom_loss(ima_net_batch, ima_ref_batch, p.LOSS_NICKNAME)
      mse = torch.mean((ima_net_batch - ima_ref_batch) ** 2)   
      # maxval = torch.max(ima_ref_batch) - torch.min(ima_ref_batch)
      # psnr = 20 * torch.log10(maxval) - 10 * torch.log10(mse)
      # ssim_value = ssim(ima_net_batch, ima_ref_batch, data_range=ima_net_batch.max() - ima_net_batch.min(), size_average=True)
      ssim_value = compare_ssim(np.array(ima_net_batch.cpu().detach().squeeze()), np.array(ima_ref_batch.cpu().detach().squeeze()),win_size=None, gradient=False, data_range=1, channel_axis=0, multichannel=False, gaussian_weights=False, full=False)
      psnr = compare_psnr(np.array(ima_net_batch.cpu().detach().squeeze()), np.array(ima_ref_batch.cpu().detach().squeeze()), data_range=1)
      # Keep track of the calculated loss
      batch_size = sinogram_batch.shape[0]
      val_loss_accumulator.update_losses(batch_size, loss.item()*batch_size)
      val_index_accumulator.update_index(batch_size, mse.item()*batch_size, psnr.item()*batch_size, ssim_value.item()*batch_size)
      # Log the current batch
      log_batch_to_console_tensorboard_harddrive(
        id_epoch, 'Val', val_loss_accumulator, time_epoch, id_batch, nb_batch_val, ima_ref_batch, ima_net_batch,
        file_name_batch, writer, flip_val, p, print_epoch_dir)

  # Save current network checkpoint if new minimal validation loss is found
  current_val_loss = val_loss_accumulator.get_epoch_loss()
  current_val_mse, current_val_psnr, current_val_ssim = val_index_accumulator.get_epoch_index()
  # if current_val_loss < min_val_loss and current_val_psnr > max_val_psnr and current_val_ssim > max_val_ssim :
  #   min_val_loss = current_val_loss
  #   max_val_psnr = current_val_psnr
  #   max_val_ssim = current_val_ssim
  #   new_val_loss = True
  #   checkpoint_file_path_and_name =  os.path.join(model_checkpoint_output_dir, f'epoch_{id_epoch}_val_loss_{min_val_loss*10:.5f}_psnr_{max_val_psnr:.3f}_ssim_{max_val_ssim:.3f}.pt')
  #   torch.save({
  #     'epoch': id_epoch +1,
  #     'network_state_dict': network.state_dict(),
  #     'optimizer_state_dict': optimizer.state_dict(),
  #     'scheduler_state_dict': scheduler.state_dict()},
  #     checkpoint_file_path_and_name
  #     )
    # loss_valtst.append(current_val_loss)
    # psnr_valtst.append(current_val_psnr)
    # ssim_valtst.append(current_val_ssim)
    # mse_valtst.append(current_val_mse)
  checkpoint_file_path_and_name =  os.path.join(model_checkpoint_output_dir, f'epoch_{id_epoch}_val_loss_{min_val_loss*10:.5f}.pt')
  if current_val_loss < min_val_loss :
      min_val_loss = current_val_loss
      new_val_loss = True
      torch.save({
        'epoch': id_epoch +1,
        'network_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()},
        #checkpoint_file_path_and_name
        checkpoint_file_path_and_name
        )
  #     loss_valtst.append(current_val_loss)
  #     psnr_valtst.append(current_val_psnr)
  #     ssim_valtst.append(current_val_ssim)
  #     mse_valtst.append(current_val_mse)
  else:
    new_val_loss = False
  # current_val_mse, current_val_psnr, current_val_ssim = val_index_accumulator.get_epoch_index()
  loss_val.append(current_val_loss)
  psnr_val.append(current_val_psnr)
  ssim_val.append(current_val_ssim)
  mse_val.append(current_val_mse)
  # Log the current epoch
  log_epoch_console_tensorboard(
    writer, 'Val', '1_validation_loss', id_epoch, time_epoch, current_val_loss,current_val_psnr, current_val_ssim, new_val_loss)
  log_twoscalar_compare(
    writer,'1_trn_val_loss_compare','train','val', id_epoch, current_train_loss,current_val_loss)
  log_twoscalar_compare(
    writer,'1_trn_val_psnr_compare','train','val', id_epoch, current_train_psnr,current_val_psnr)
  log_twoscalar_compare(
    writer,'1_trn_val_ssim_compare','train','val', id_epoch, current_train_ssim,current_val_ssim)
  #return current_val_loss, current_val_mse, current_val_psnr, current_val_ssim

# ================================================================
# === Testing routine ============================================
# ================================================================
def tst(p, network, loader_tst, nb_batch_tst, id_epoch, device, print_epoch_dir, writer):

  global new_val_loss
  global last_test_loss
  global flip_tst
  global loss_tst
  global psnr_tst
  global mse_tst
  global ssim_tst
  global last_test_psnr
  global last_test_ssim
  # global current_tst_loss, current_tst_mse, current_tst_psnr, current_tst_ssim
  time_epoch = time.time()

  if new_val_loss: 
    # 意思是当val的loss变小之后再进行测试，这也就是test有好多轮都没有数据的原因
    network.eval()  # network.eval() 是PyTorch中用于将神经网络模型设置为评估模式的方法。在评估模式下，模型会关闭训练特定层（如Dropout层），并且不会计算梯度，这在推断阶段是非常有用的。
    # Dropout层是一种常用的正则化技术，用于防止神经网络过拟合。
    # 在训练过程中，Dropout层会随机地将一部分神经元的输出设置为0，从而减少神经元之间的依赖关系，有助于提高模型的泛化能力和减少过拟合的风险
    tst_loss_accumulator = EpochLossAccumulator()
    tst_index_accumulator = EpochIndexAccumulator()
    with torch.no_grad():
      for id_batch, (sinogram_batch, ima_ref_batch, file_name_batch) in enumerate(loader_tst):
        # Forward pass with the deep neural network
        sos = torch.tensor([1476,1512,1548,1584,1620], device = device).float().repeat(sinogram_batch.shape[0])\
              .view(sinogram_batch.shape[0],5).T
        ima_net_batch = network(sinogram_batch.to(device).float(), sos)
        # Loss
        ima_ref_batch = ima_ref_batch.to(device).float()
        loss = custom_loss(ima_net_batch, ima_ref_batch, p.LOSS_NICKNAME)
        mse = torch.mean((ima_net_batch - ima_ref_batch) ** 2)   
        # maxval = torch.max(ima_ref_batch) - torch.min(ima_ref_batch)
        # psnr = 20 * torch.log10(maxval) - 10 * torch.log10(mse)
        # ssim_value = ssim(ima_net_batch, ima_ref_batch, data_range=ima_net_batch.max() - ima_net_batch.min(), size_average=True)
        ssim_value = compare_ssim(np.array(ima_net_batch.cpu().detach().squeeze()), np.array(ima_ref_batch.cpu().detach().squeeze()),win_size=None, gradient=False, data_range=1, channel_axis=0, multichannel=False, gaussian_weights=False, full=False)
        psnr = compare_psnr(np.array(ima_net_batch.cpu().detach().squeeze()), np.array(ima_ref_batch.cpu().detach().squeeze()), data_range=1)
        # Keep track of the calculated loss
        batch_size = sinogram_batch.shape[0]
        tst_loss_accumulator.update_losses(batch_size, loss.item()*batch_size)
        tst_index_accumulator.update_index(batch_size, mse.item()*batch_size, psnr.item()*batch_size, ssim_value.item()*batch_size)
        # Log the current batch
        log_batch_to_console_tensorboard_harddrive(
          id_epoch, 'Tst', tst_loss_accumulator, time_epoch, id_batch, nb_batch_tst, ima_ref_batch, ima_net_batch,
          file_name_batch, writer, flip_tst, p, print_epoch_dir)

    # Keep track of last_test_loss
    printed_test_loss = tst_loss_accumulator.get_epoch_loss()
    last_test_loss = printed_test_loss
    current_tst_mse, current_tst_psnr, current_tst_ssim = tst_index_accumulator.get_epoch_index()
    last_test_psnr = current_tst_psnr
    last_test_ssim = current_tst_ssim
    log_twoscalar_compare(
      writer,'2_val_tst_loss_compare','val','tst', id_epoch, current_val_loss, printed_test_loss)
    log_twoscalar_compare(
      writer,'2_val_tst_psnr_compare','val','tst', id_epoch, current_val_psnr,current_tst_psnr)
    log_twoscalar_compare(
      writer,'2_val_tst_ssim_compare','val','tst', id_epoch, current_val_ssim,current_tst_ssim)
    #loss_tst.append(printed_test_loss)
    # psnr_tst.append(current_tst_psnr)
    # mse_tst.append(current_tst_mse)
    # ssim_tst.append(current_tst_ssim)
    loss_valtst.append(current_val_loss)
    psnr_valtst.append(current_val_psnr)
    ssim_valtst.append(current_val_ssim)
    loss_tst.append(printed_test_loss)
    psnr_tst.append(current_tst_psnr)
    mse_tst.append(current_tst_mse)
    ssim_tst.append(current_tst_ssim)
    log_epoch_console_tensorboard(
      writer, 'Tst', '2_testing_loss', id_epoch, time_epoch, printed_test_loss, current_tst_psnr, current_tst_ssim, new_val_loss)
  else:

    # This dummy text in the console will not be seen, but is a necessary placeholder for the log display
    print('(The test set is not processed)')
    printed_test_loss = last_test_loss
    log_epoch_console_tensorboard(
      writer, 'Tst', '2_testing_loss', id_epoch, time_epoch, printed_test_loss, last_test_psnr, last_test_ssim, new_val_loss)

  # Log the current epoch
  # log_epoch_console_tensorboard(
  #   writer, 'Tst', '2_testing_loss', id_epoch, time_epoch, printed_test_loss, current_tst_psnr, current_tst_ssim, new_val_loss)

  #return printed_test_loss, current_tst_mse, current_tst_psnr, current_tst_ssim
def set_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True