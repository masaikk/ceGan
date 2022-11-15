#
# Copyright (C), 2018-上午10:30
# FileName: cegan_instructor.py
# Author:   b8313
# Date:     上午10:30 上午10:30
# Description: 
#
import copy
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from metrics.nll import NLL
from models.CeGAN_G import CeGAN_G
from models.CeGAN_D_sep import CeGAN_vec_D, CeGAN_sen_D
from utils.helpers import get_fixed_temperature, get_losses, get_format_time
from utils.data_loader import GenDataIter, DisDataIter
from utils.gan_loss import GANLoss
from utils.text_process import tensor_to_tokens


class CeGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(CeGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = CeGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                           cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.parents = [CeGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                                cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA).state_dict()
                        for _ in range(cfg.n_parent)]
        # TODO :realize why use this 'self.parents'
        # self.dis = CeGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size, cfg.padding_idx,
        #                    gpu=cfg.CUDA)
        # I will not use single discriminator

        self.vec_dis = CeGAN_vec_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size, cfg.padding_idx,
                                   gpu=cfg.CUDA)
        self.sen_dis = CeGAN_sen_D(cfg.dis_embed_dim, cfg.vocab_size, cfg.padding_idx, gpu=cfg.CUDA)

        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.vec_dis_opt = optim.Adam(self.vec_dis.parameters(), lr=cfg.dis_lr)
        # for word embedding discriminator
        self.sen_dis_opt = optim.Adam(self.sen_dis.parameters(), lr=cfg.dis_lr)
        # for sentence discriminator

        self.parent_mle_opts = [copy.deepcopy(self.gen_opt.state_dict())
                                for _ in range(cfg.n_parent)]
        self.parent_adv_opts = [copy.deepcopy(self.gen_adv_opt.state_dict())
                                for _ in range(cfg.n_parent)]  # list of optimizer state dict

        # Criterion
        self.G_criterion = [GANLoss(loss_mode, 'G', cfg.d_type, CUDA=cfg.CUDA) for loss_mode in cfg.mu_type.split()]
        self.D_criterion = GANLoss(cfg.loss_type, 'D', cfg.d_type, CUDA=cfg.CUDA)

        # init_model & load_gen was copied from evo-gan

    # re-write from basic instroctor
    def init_model(self):
        if cfg.dis_pretrain:
            self.log.info(
                'pretrain/image_coco/dis_pretrain_cegan_RMC_sl37_sn10000_time202110291728_batch32_eval8.pt')
            self.sen_dis.load_state_dict(
                torch.load('pretrain/image_coco/dis_pretrain_cegan_RMC_sl37_sn10000_time202110291728_batch32_eval8.pt'))
            # self.log.info(
            #     'Load pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))
            # self.vec_dis.load_state_dict(torch.load(cfg.pretrained_dis_path, map_location='cuda:{}'.format(cfg.device)))
        if cfg.gen_pretrain:
            self.log.info(
                'pretrain/image_coco/gen_MLE_pretrain_cegan_RMC_sl37_sn10000_epoch30_time202110291728_batch32_eval8.pt0')
            self.gen.load_state_dict(torch.load(
                'pretrain/image_coco/gen_MLE_pretrain_cegan_RMC_sl37_sn10000_epoch30_time202110291728_batch32_eval8.pt0'))
            # self.log.info('Load MLE pre-trained generator: {}'.format(cfg.pretrained_gen_path))
            # self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path, map_location='cuda:{}'.format(cfg.device)))

        if cfg.CUDA:
            self.gen = self.gen.cuda()
            self.vec_dis = self.vec_dis.cuda()
            self.sen_dis = self.sen_dis.cuda()

    def load_gen(self, parent, parent_opt, mle=False):
        self.gen.load_state_dict(copy.deepcopy(parent))
        if mle:
            self.gen_opt.load_state_dict(copy.deepcopy(parent_opt))
            self.gen_opt.zero_grad()
        else:
            self.gen_adv_opt.load_state_dict(copy.deepcopy(parent_opt))
            self.gen_adv_opt.zero_grad()

    def _run(self):
        # ===PRE-TRAINING (GENERATOR)===
        if not cfg.gen_pretrain:
            # self.log.info('Starting Generator MLE Training...')
            # self.pretrain_generator(cfg.MLE_train_epoch)
            # if cfg.if_save and not cfg.if_test:
            #     torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
            #     print('Save pretrain_generator: {}'.format(cfg.pretrained_gen_path))

            for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_mle_opts)):
                self.log.info('Starting Generator-{} MLE Training...'.format(i))
                self.load_gen(parent, parent_opt, mle=True)  # load state dict
                self.pretrain_generator(cfg.MLE_train_epoch)
                self.parents[i] = copy.deepcopy(self.gen.state_dict())  # save state dict
                if cfg.if_save and not cfg.if_test:
                    torch.save(self.gen.state_dict(), cfg.pretrained_gen_path + '%d' % i)
                    self.log.info('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path + '%d' % i))

        # TODO: In coding, pre-train generator will not be used

        # # ===TRAIN DISCRIMINATOR     only for sentence discriminator ====
        if not cfg.dis_pretrain:
            self.log.info('Starting sentence  Discriminator Training...')
            self.train_discriminator(cfg.d_step, cfg.d_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.sen_dis.state_dict(), cfg.pretrained_dis_path)
                print('Save pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))

        # from seq-gan, pre-train discriminator
        #         if not cfg.dis_pretrain:
        #             self.log.info('Starting Discriminator Training...')
        #             self.train_discriminator(cfg.d_step, cfg.d_epoch)
        #             if cfg.if_save and not cfg.if_test:
        #                 torch.save(self.sen_dis.state_dict(), cfg.pretrained_dis_path)
        #                 print('Save pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))

        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        progress = tqdm(range(cfg.ADV_train_epoch))
        # for adv_epoch in progress:
        # self.sig.update()
        # if self.sig.adv_sig:
        #     g_loss = self.adv_train_generator(cfg.ADV_g_step)  # Generator
        #     d_loss = self.adv_train_discriminator(cfg.ADV_d_step)  # Discriminator
        #     self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature
        #
        #     progress.set_description(
        #         'g_loss: %.4f, d_loss: %.4f, temperature: %.4f' % (g_loss, d_loss, self.gen.temperature))
        for adv_epoch in progress:
            if cfg.temperature == 1:
                score, fit_score, select_mu = self.evolve_generator(cfg.ADV_g_step)
            else:  # evolve with temperature
                score, fit_score, select_mu = self.evolve_generator_with_temp(adv_epoch, cfg.ADV_g_step)
            d_loss = self.evolve_discriminator(cfg.ADV_d_step)

            best_id = int(np.argmax(score))
            progress.set_description('mu: %s, d_loss = %.4f, temp = %.4f' % (
                ' '.join(select_mu), d_loss, self.parents[best_id]['temperature'].item()))

            # TEST
            if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1 and adv_epoch!=0:
                best_id = int(np.argmax(score))
                self.load_gen(self.parents[best_id], self.parent_adv_opts[best_id])

                self.log.info('[ADV] %s epoch %d: temp = %.4f, d_loss = %.4f, %s' % (
                    get_format_time(), adv_epoch, self.gen.temperature.item(), d_loss, self.cal_metrics(fmt_str=True)))

                if cfg.if_save and not cfg.if_test:
                    self._save('ADV', adv_epoch)

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                # ===Train===
                pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)

                # ===Test===
                if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                    self.log.info('[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (
                        epoch, pre_loss, self.cal_metrics(fmt_str=True)))

                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break

    def train_discriminator(self, d_step, d_epoch, phase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.

        this is use for sen_dis
        """
        # prepare loader for validate
        global d_loss, train_acc
        for step in range(d_step):
            # prepare loader for training
            pos_samples = self.train_data.target
            neg_samples = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
            dis_data = DisDataIter(pos_samples, neg_samples)

            for epoch in range(d_epoch):
                # ===Train===
                d_loss, train_acc = self.train_dis_epoch(self.sen_dis, dis_data.loader, self.dis_criterion,
                                                         self.sen_dis_opt)

            # ===Test===
            self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f,' % (
                phase, step, d_loss, train_acc))

            if cfg.if_save and not cfg.if_test:
                torch.save(self.sen_dis.state_dict(), cfg.pretrained_dis_path)

    # This function is copied from mali-gan
    # def get_mali_reward(self, samples):
    #     rewards = []
    #     for _ in range(cfg.rollout_num):
    #         dis_out = F.softmax(self.sen_dis(samples), dim=-1)[:, 1]
    #         rewards.append(dis_out)
    #
    #     rewards = torch.mean(torch.stack(rewards, dim=0), dim=0)  # batch_size
    #     rewards = torch.div(rewards, 1 - rewards)
    #     rewards = torch.div(rewards, torch.sum(rewards))
    #     rewards -= torch.mean(rewards)
    #     rewards = rewards.unsqueeze(1).expand(samples.size())  # batch_size * seq_len
    #
    #     return rewards

    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # ===Train===
            d_out_real = self.vec_dis(real_samples)
            d_out_fake = self.vec_dis(gen_samples)
            g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    # these seven functions were copied from evo-gan
    def evolve_generator(self, evo_g_step):
        # evaluation real data
        self.prepare_eval_real_data()

        best_score = np.zeros(cfg.n_parent)
        best_fit = []
        best_child = []
        best_child_opt = []
        best_fake_samples = []
        selected_mutation = []
        count = 0

        # all children share the same real data output from Discriminator
        with torch.no_grad():
            real_samples = F.one_hot(self.train_data.random_batch()['target'], cfg.vocab_size).float()
            if cfg.CUDA:
                real_samples = real_samples.cuda()
            self.d_out_real = self.vec_dis(real_samples)

        for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_adv_opts)):
            for j, criterionG in enumerate(self.G_criterion):
                # Variation
                self.load_gen(parent, parent_opt)  # load state dict to self.gen
                # single loss
                self.variation(evo_g_step, criterionG)

                # Evaluation
                self.prepare_eval_fake_data()  # evaluation fake data
                Fq, Fd, score = self.evaluation(cfg.eval_type)

                # Selection
                if count < cfg.n_parent:
                    best_score[count] = score
                    best_fit.append([Fq, Fd, score])
                    best_child.append(copy.deepcopy(self.gen.state_dict()))
                    best_child_opt.append(copy.deepcopy(self.gen_adv_opt.state_dict()))
                    best_fake_samples.append(self.eval_fake_samples)
                    selected_mutation.append(criterionG.loss_mode)
                else:  # larger than previous child, replace it
                    fit_com = score - best_score
                    if max(fit_com) > 0:
                        id_replace = np.where(fit_com == max(fit_com))[0][0]
                        best_score[id_replace] = score
                        best_fit[id_replace] = [Fq, Fd, score]
                        best_child[id_replace] = copy.deepcopy(self.gen.state_dict())
                        best_child_opt[id_replace] = copy.deepcopy(self.gen_adv_opt.state_dict())
                        best_fake_samples[id_replace] = self.eval_fake_samples
                        selected_mutation[id_replace] = criterionG.loss_mode
                count += 1

        self.parents = copy.deepcopy(best_child)
        self.parent_adv_opts = copy.deepcopy(best_child_opt)
        self.best_fake_samples = torch.cat(best_fake_samples, dim=0)
        return best_score, np.array(best_fit), selected_mutation

    def evolve_generator_with_temp(self, cur_adv_step, evo_g_step):
        # evaluation real data
        self.prepare_eval_real_data()

        best_score = np.zeros(cfg.n_parent)
        best_fit = []
        best_child = []
        best_child_opt = []
        best_fake_samples = []
        selected_mutation = []
        count = 0
        self.temp_log.log(logging.INFO, '{} Epoch {} train...'.format(get_format_time(), cur_adv_step))

        # all children share the same real data output from Discriminator
        with torch.no_grad():
            real_samples = F.one_hot(self.train_data.random_batch()['target'], cfg.vocab_size).float()
            if cfg.CUDA:
                real_samples = real_samples.cuda()
            self.d_out_real_vec = self.vec_dis(real_samples)
            self.d_out_real_sen = self.sen_dis(torch.argmax(real_samples, dim=2))

        for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_adv_opts)):
            for j, criterionG in enumerate(self.G_criterion):
                all_temp, info_to_log = self.get_evo_temp(cur_adv_step)  # get evo temp
                self.temp_log.log(logging.INFO, info_to_log)
                all_temp_with_score = []
                temp_score = float('-inf')
                temp_fit = None
                temp_child = None
                temp_child_opt = None
                temp_fake_samples = None
                best_child_index = 0

                # Selection based on temperature, use eval_type=nll
                for temp_index in range(len(all_temp)):
                    temp = all_temp[temp_index]
                    # Variation
                    self.load_gen(parent, parent_opt)  # load state dict to self.gen
                    self.gen.temperature.data = temp  # update Generator temperature

                    self.variation(evo_g_step, criterionG)

                    # Evaluation
                    self.prepare_eval_fake_data()  # evaluation fake data
                    # print('epoch {} to evolution temp'.format(evo_g_step))
                    # _, _, t_score = self.evaluation('Ra')  # for temp evolutionary
                    print('\n\nEpoch {}, criterionG {} to evolution loss...'.format(cur_adv_step, j))
                    # self.temp_log.log(logging.INFO,
                    #     '\n{} Epoch {}, criterionG {} to evolution loss...'.format(get_format_time(), cur_adv_step, j))
                    loss_Fq, loss_Fd, loss_score = self.evaluation(cfg.eval_type)  # for loss evolutionary
                    t_score = loss_score
                    all_temp_with_score.append([temp.item(), t_score])
                    if t_score > temp_score:
                        temp_score = loss_score
                        temp_fit = [loss_Fq, loss_Fd, loss_score]
                        temp_child = copy.deepcopy(self.gen.state_dict())
                        temp_child_opt = copy.deepcopy(self.gen_adv_opt.state_dict())
                        temp_fake_samples = copy.deepcopy(self.eval_fake_samples)
                        best_child_index = temp_index
                        # print('       This epoch temperature:' + str(temp.item()))
                        # self.temp_log.log(logging.INFO, '       This epoch temperature:' + str(temp.item()))
                self.temp_log.log(logging.INFO,
                                  ' This epoch temperature: {} with score: {} with child_index as {}'.format(
                                      all_temp_with_score[best_child_index][0],
                                      all_temp_with_score[best_child_index][1], best_child_index))

                print('             This epoch all temperature with score are {}'.format(str(all_temp_with_score)))
                self.temp_log.log(logging.INFO,
                                  '{} This epoch {} all temperature with score are {}'.format(get_format_time(),
                                                                                              cur_adv_step,
                                                                                              str(all_temp_with_score)))
                # all_temp_with_score=[]
                # Selection based on mu_type, use eval_type=cfg.eval_type
                if count < cfg.n_parent:
                    best_score[count] = temp_score
                    best_fit.append(temp_fit)
                    best_child.append(temp_child)
                    best_child_opt.append(temp_child_opt)
                    best_fake_samples.append(temp_fake_samples)
                    selected_mutation.append(criterionG.loss_mode)
                else:  # larger than previous child, replace it
                    fit_com = temp_score - best_score
                    if max(fit_com) > 0:
                        id_replace = np.where(fit_com == max(fit_com))[0][0]
                        best_score[id_replace] = temp_score
                        best_fit[id_replace] = temp_fit
                        best_child[id_replace] = temp_child
                        best_child_opt[id_replace] = temp_child_opt
                        best_fake_samples[id_replace] = temp_fake_samples
                        selected_mutation[id_replace] = criterionG.loss_mode
                count += 1

        self.parents = copy.deepcopy(best_child)
        self.parent_adv_opts = copy.deepcopy(best_child_opt)
        self.best_fake_samples = torch.cat(best_fake_samples, dim=0)
        return best_score, np.array(best_fit), selected_mutation

    def evolve_discriminator(self, evo_d_step):
        total_loss = 0
        for step in range(evo_d_step):
            real_samples = F.one_hot(self.train_data.random_batch()['target'], cfg.vocab_size).float()
            gen_samples = self.best_fake_samples[step * cfg.batch_size:(step + 1) * cfg.batch_size]
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()

            # ===Train===
            d_out_real_vec = self.vec_dis(real_samples)
            d_out_fake_vec = self.vec_dis(gen_samples)
            d_loss_vec = self.D_criterion(d_out_real_vec, d_out_fake_vec)
            print('   This evolve_discriminator turn d_loss_vec=', str(d_loss_vec.item()))

            self.optimize(self.vec_dis_opt, d_loss_vec, self.vec_dis)

            d_out_real_sen = self.sen_dis(torch.argmax(real_samples, dim=2))
            d_out_fake_sen = self.sen_dis(torch.argmax(gen_samples, dim=2))
            d_loss_sen = self.D_criterion(d_out_real_sen, d_out_fake_sen)
            print('   This evolve_discriminator turn d_loss_sen=', str(d_loss_sen.item()))

            self.optimize(self.sen_dis_opt, d_loss_sen, self.sen_dis)

            # now, self.vec_dis_opt in evo-gan was been seen as self.vec_dis_opt
            # TODO: fix this value to fit two discriminators in ce-gan
            total_loss += cfg.vec_dis_fact * d_loss_vec.item() + (1 - cfg.vec_dis_fact) * d_loss_sen.item()
            # total_loss += d_loss_vec.item()
            print("          evolve_discriminator TOTAL LOSS : {}".format(str(total_loss)))

        return total_loss / evo_d_step if evo_d_step != 0 else 0

    def evaluation(self, eval_type):
        """Evaluation all children, update child score. Note that the eval data should be the same"""
        eval_samples = self.gen.sample(cfg.eval_b_num * cfg.batch_size, cfg.max_bn * cfg.batch_size)
        gen_data = GenDataIter(eval_samples)

        # Fd
        if cfg.lambda_fd != 0:
            Fd = NLL.cal_nll(self.gen, gen_data.loader, self.mle_criterion)  # NLL_div
        else:
            Fd = 0

        # Fq
        if eval_type == 'standard':
            Fq = self.eval_d_out_fake_vec.mean().cpu().item()
        elif eval_type == 'rsgan':
            g_loss_vec, d_loss = get_losses(self.eval_d_out_real_vec, self.eval_d_out_fake_vec, 'rsgan')
            Fq = d_loss.item()
        elif 'bleu' in eval_type:
            self.bleu.reset(test_text=tensor_to_tokens(eval_samples, self.idx2word_dict))

            if cfg.lambda_fq != 0:
                Fq = self.bleu.get_score(given_gram=int(eval_type[-1]))
            else:
                Fq = 0
        elif 'Ra' in eval_type:
            g_loss_vec = torch.sigmoid(self.eval_d_out_fake_vec - torch.mean(self.eval_d_out_real_vec)).sum() / (
                self.eval_d_out_real_vec.shape[0])
            g_loss_sen = torch.sigmoid(self.eval_d_out_fake_sen - torch.mean(self.eval_d_out_real_sen)).sum() / (
                    self.eval_d_out_real_sen.shape[0] * self.eval_d_out_real_sen.shape[1])
            print('        g_loss_vec is {} and g_loss_sen is {}'.format(g_loss_vec.item(), g_loss_sen.item()))
            Fq = cfg.vec_dis_fact * g_loss_vec.item() + (1 - cfg.vec_dis_fact) * g_loss_sen.item()
        else:
            raise NotImplementedError("Evaluation '%s' is not implemented" % eval_type)

        score = cfg.lambda_fq * Fq + cfg.lambda_fd * Fd
        print('     This epoch Fq is {} & Fd is {}'.format(Fq, Fd))
        return Fq, Fd, score

    def prepare_eval_real_data(self):
        with torch.no_grad():
            self.eval_real_samples = torch.cat(
                [F.one_hot(self.train_data.random_batch()['target'], cfg.vocab_size).float()
                 for _ in range(cfg.eval_b_num)], dim=0)
            if cfg.CUDA:
                self.eval_real_samples = self.eval_real_samples.cuda()

            if cfg.eval_type == 'rsgan' or cfg.eval_type == 'Ra':
                self.eval_d_out_real_vec = self.vec_dis(self.eval_real_samples)

                self.eval_d_out_real_sen = self.sen_dis(torch.argmax(self.eval_real_samples, dim=2))

    def prepare_eval_fake_data(self):
        with torch.no_grad():
            self.eval_fake_samples = self.gen.sample(cfg.eval_b_num * cfg.batch_size,
                                                     cfg.eval_b_num * cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                self.eval_fake_samples = self.eval_fake_samples.cuda()

            if cfg.eval_type == 'rsgan' or cfg.eval_type == 'Ra':
                self.eval_d_out_fake_vec = self.vec_dis(self.eval_fake_samples)
                self.eval_d_out_fake_sen = self.sen_dis(torch.argmax(self.eval_fake_samples, dim=2))

    def variation(self, g_step, criterionG):
        """
        Must call self.load_gen() before variation
        :param g_step:
        :param criterionG:
        :return:
        """
        total_loss = 0
        for step in range(g_step):
            # real_samples = F.one_hot(self.train_data.random_batch()['target'], cfg.vocab_size).float()
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                # real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
                gen_samples = gen_samples.cuda()

            # ===Train===
            # d_out_real = self.vec_dis(real_samples)
            d_out_fake_vec = self.vec_dis(gen_samples)
            d_out_fake_sen = self.sen_dis(torch.argmax(gen_samples, dim=2))
            # g_loss = criterionG(d_out_real, d_out_fake)

            g_loss_vec = criterionG(self.d_out_real_vec, d_out_fake_vec)
            g_loss_sen = criterionG(self.d_out_real_sen, d_out_fake_sen)
            g_loss = cfg.vec_dis_fact * g_loss_vec + (1 - cfg.vec_dis_fact) * g_loss_sen
            print('\nVariation turn g_loss_vec is {}, g_loss_sen is {} and g_loss is {}'.format(g_loss_vec, g_loss_sen,
                                                                                                g_loss))
            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    @staticmethod
    def get_evo_temp(cur_step):
        """

        :param cur_step:
        :return:
        """

        """randomly get different temperature according to current adversarial step"""
        fn_mu_temp_type = cfg.fn_mu_temp.split()
        mu_temp_type = cfg.mu_temp.split()
        all_temp = list()
        temp_info_to_log = ''
        temp_n, temp_log_n = get_fixed_temperature(cfg.temperature, cur_step, cfg.ADV_train_epoch,
                                                   random.choice(fn_mu_temp_type))
        # all_temp.append(get_fixed_temperature(1.0, 0, 0, 'no'))  # temp=1.0
        all_temp.append(temp_n)  # current step
        temp_info_to_log += temp_log_n

        temp_n1, temp_log_n1 = get_fixed_temperature(cfg.temperature, cur_step + cfg.evo_temp_step, cfg.ADV_train_epoch,
                                                     random.choice(mu_temp_type))
        all_temp.append(temp_n1)
        temp_info_to_log += temp_log_n1

        if cur_step > cfg.evo_temp_step:
            temp_n2, temp_log_n2 = get_fixed_temperature(cfg.temperature, cur_step - cfg.evo_temp_step,
                                                         cfg.ADV_train_epoch,
                                                         random.choice(mu_temp_type))
            all_temp.append(temp_n2)
            temp_info_to_log += temp_log_n2

        return torch.Tensor(all_temp), temp_info_to_log  # three temp
