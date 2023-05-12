import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from data import cfg_re50
from models.retinaface import RetinaFace


class FaceDetector():

    def __init__(self):
        trained_model = './weights/Resnet50_Final.pth'
        self.net = self._init_network(trained_model)
        self.device = torch.device("cuda")


    def run(self, image):
        pass


    def show_results(self, image, bboxes):
        pass


    def _preprocess_image(self, image):
        image = np.float32(image)
        image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(self.device)


        return image

    def _init_network(self, trained_model=None):
        net = RetinaFace(cfg=cfg_re50, phase = 'test')
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


    def _load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model


if __name__ == '__main__':

    face_detector = FaceDetector()
    image = cv2.imread('test.jpg')

    bboxes = face_detector.run(image)
    face_detector.show_results(image, bboxes)

