import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from data import cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm


class FaceDetector():

    def __init__(self):
        trained_model = './weights/Resnet50_Final.pth'
        self.cfg = cfg_re50
        self.confidence_threshold = 0.02
        self.nms_threshold = 0.4
        self.vis_thres = 0.5
        self.device = torch.device("cuda")

        self.net = self._init_network(trained_model)


    def run(self, image):
        image_height, image_width, _ = image.shape
        scale = torch.Tensor([image_width, image_height, image_width, image_height])
        scale = scale.to(self.device)
        preprocessed_image = self._preprocess_image(image)
        loc, conf, _ = self.net(preprocessed_image)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(image_height, image_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]

        return dets


    def show_results(self, image, bboxes):
        for b in bboxes:
            if b[4] < self.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        cv2.imshow('test', image)
        cv2.waitKey(0)

    def _preprocess_image(self, image):
        image = np.float32(image)
        image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(self.device)


        return image

    def _init_network(self, trained_model=None):
        net = RetinaFace(cfg=self.cfg, phase = 'test')
        if trained_model is not None:
            net = self._load_model(net, trained_model)
        net.eval()
        cudnn.benchmark = True
        net = net.to(self.device)
        return net


    def _check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def _remove_prefix(self, state_dict, prefix):
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def _load_model(self, model, pretrained_path):
        print('Loading pretrained model from {}'.format(pretrained_path))
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, 'module.')
        self._check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model


if __name__ == '__main__':

    face_detector = FaceDetector()
    image = cv2.imread('test.jpg')

    bboxes = face_detector.run(image)
    face_detector.show_results(image, bboxes)

