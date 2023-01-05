# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/12/22 9:56 上午
==================================="""
# a flask api return a mp4 file
import os
from os.path import join as pjoin

import spacy
import torch
from flask import Flask, send_file, request, jsonify, Response
import sys
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
from torch.utils.data import DataLoader

from data.dataset import Motion2TextEvalDataset
from networks.modules import Seq2SeqText2MotModel, VQDecoderV3
from networks.quantizer import Quantizer, EMAVectorQuantizer
from options.evaluate_options import TestT2MOptions
from utils import paramUtil
from utils.plot_script import plot_3d_motion
from utils.utils import motion_temporal_filter
from utils.word_vectorizer import WordVectorizerV2

from scripts.motion_process import recover_from_ric

app = Flask(__name__)


def plot_t2m(data, captions, save_dir):
    data = data * std + mean  # (1 ,192, 263)
    for i in range(len(data)):
        joint_data = data[i]
        caption = captions[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        joint = motion_temporal_filter(joint)
        save_path = '%s_%02d.mp4' % (save_dir, i)
        np.save('%s_%02d.npy' % (save_dir, i),
                {'motion': [data], 'text': [captions], 'lengths': [len(data)],
                 'num_samples': 1, 'num_repetitions': 1})

        # np.save('%s_%02d.npy' % (save_dir, i), joint)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius)
        return save_path


class DataProcess:
    def __init__(self, opt, mean, std, w_vectorizer, nlp):
        self.opt = opt
        self.mean = mean
        self.std = std
        self.w_vectorizer = w_vectorizer
        self.nlp = nlp

    def process(self, data):
        sentence = data.strip()
        doc = self.nlp(sentence)

        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue

            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        data = data * self.std + self.mean
        return data

    def embeded(self, sentence):
        word_list, pos_list = self.process(sentence.strip())
        tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]
        if len(tokens) < self.opt.max_text_len:
            tokens += ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.opt.max_text_len]
            sent_len = len(tokens)
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh, _ = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, [sentence.strip()], torch.tensor([sent_len])


def get_chunk(file_name, start, end):
    file_size = os.path.getsize(file_name)
    start = 0 if start is None else int(start)
    length = file_size - start if end is None else int(end) - start

    with open(file_name, 'rb') as f:
        f.seek(start)
        data = f.read(length)
        return data, start, length, file_size


def build_models(opt, dec_channels):
    vq_decoder = VQDecoderV3(opt.dim_vq_latent, dec_channels, opt.n_resblk, opt.n_down)
    quantizer = None
    if opt.q_mode == 'ema':
        quantizer = EMAVectorQuantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)
    elif opt.q_mode == 'cmt':
        quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'model', 'finest.tar'),
                            map_location=opt.device)
    vq_decoder.load_state_dict(checkpoint['vq_decoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])

    t2m_model = Seq2SeqText2MotModel(300, n_mot_vocab, opt.dim_txt_hid, opt.dim_mot_hid,
                                     opt.n_mot_layers, opt.device, opt.early_or_late)
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'finest.tar'),
                            map_location=opt.device)
    t2m_model.load_state_dict(checkpoint['t2m_model'])
    print('Loading t2m_model model: Epoch %03d Total_Iter %03d' % (checkpoint['ep'], checkpoint['total_it']))

    return vq_decoder, quantizer, t2m_model


@app.route('/t2video', methods=['POST'])
def video():
    params = request.get_json()
    text = params.get('text', None)
    if text is None:
        return jsonify({'status': 'error', 'msg': 'text is None'}), 400

    vq_decoder.to(opt.device)
    quantizer.to(opt.device)
    t2m_model.to(opt.device)

    vq_decoder.eval()
    quantizer.eval()
    t2m_model.eval()
    word_emb, pos_one_hots, captions, cap_lens = data_process.embeded(text)
    word_emb = torch.from_numpy(word_emb).float().unsqueeze(0).to(opt.device)
    pos_one_hots = torch.from_numpy(pos_one_hots).float().unsqueeze(0).to(opt.device)
    cap_lens = cap_lens.to(opt.device)

    pred_tokens, len_map = t2m_model.sample_batch(word_emb, cap_lens, trg_sos=opt.mot_start_idx,
                                                  trg_eos=opt.mot_end_idx, max_steps=49, top_k=opt.top_k)
    pred_tokens = pred_tokens[:, 1:len_map[0] + 1]
    vq_latent = quantizer.get_codebook_entry(pred_tokens)
    gen_motion = vq_decoder(vq_latent)
    sub_dict = {}
    sub_dict['motion'] = gen_motion.cpu().detach().numpy()
    sub_dict['length'] = len(gen_motion[0])

    joint_save_path = pjoin(opt.joint_dir, 'joint')
    animation_save_path = pjoin(opt.animation_dir, 'animation')

    os.makedirs(joint_save_path, exist_ok=True)
    os.makedirs(animation_save_path, exist_ok=True)
    file_name = plot_t2m(sub_dict['motion'], captions,
                         pjoin(animation_save_path, 'gen_motion_L%03d' % (sub_dict['motion'].shape[1])))

    data, start, length, file_size = get_chunk(file_name, 0, 0)
    current_path = os.getcwd()
    data_path = os.path.join(current_path, file_name)

    return jsonify({'status': 'success', 'data': data_path}), 200


if __name__ == '__main__':
    parser = TestT2MOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.result_dir = pjoin(opt.result_path, opt.dataset_name, opt.name, opt.ext)
    opt.joint_dir = pjoin(opt.result_dir, 'joints')
    opt.animation_dir = pjoin(opt.result_dir, 'animations')
    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.m_token_dir = pjoin(opt.data_root, 'VQVAEV3_CB1024_CMT_H1024_NRES3')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_token = 55
        opt.max_motion_frame = 196
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.m_token_dir = pjoin(opt.data_root, 'VQVAEV3_CB1024_CMT_H1024_NRES3')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_token = 55
        opt.max_motion_frame = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'meta', 'std.npy'))

    n_mot_vocab = opt.codebook_size + 3
    opt.mot_start_idx = opt.codebook_size
    opt.mot_end_idx = opt.codebook_size + 1
    opt.mot_pad_idx = opt.codebook_size + 2

    enc_channels = [1024, opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, 1024, dim_pose]

    w_vectorizer = WordVectorizerV2('./text2motion/glove', 'our_vab')
    n_txt_vocab = len(w_vectorizer) + 1
    _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
    _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
    opt.txt_pad_idx = len(w_vectorizer)

    vq_decoder, quantizer, t2m_model = build_models(opt, dec_channels)

    split_file = pjoin(opt.data_root, opt.split_file)
    nlp = spacy.load('en_core_web_sm')
    data_process = DataProcess(opt, mean, std, w_vectorizer, nlp)

    app.run(debug=False, host='0.0.0.0', port=8800)
