import time
import unittest

import torch
from torch.utils._benchmark import Timer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets.build import build_dataset
from main import get_parser
from models.backbone import build_backbone
from models.criterion import build_criterion
from models.decoder import build_decoder
from models.detr import build_detr
from models.matcher import build_matcher
from models.encoder import build_encoder
from models.position import SinePositionEncoder
from utils.data import collate_fn, nested_tensor_from_image_list


class TestDataLoading(unittest.TestCase):
    def setUp(self):
        print("")

        self.args = get_parser().parse_args()
        self.globals_dict = {}

    def tearDown(self):
        torch.cuda.reset_peak_memory_stats()

    def test_coco(self):
        self.args.dataset = 'coco'

        start_loading_time = time.time()
        train_dataset, val_dataset = build_dataset(self.args)
        end_loading_time = time.time()

        load_time = end_loading_time - start_loading_time
        print(f"Load time COCO dataset: {load_time: .1f} s")

        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

        dataloader_kwargs = {'collate_fn': collate_fn, 'num_workers': self.args.num_workers, 'pin_memory': False}
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, self.args.batch_size, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **dataloader_kwargs)
        val_dataloader = DataLoader(val_dataset, self.args.batch_size, sampler=val_sampler, **dataloader_kwargs)

        train_images, train_tgt_dict = next(iter(train_dataloader))
        val_image, val_tgt_dict = next(iter(val_dataloader))

        detr = build_detr(self.args).to('cuda')
        criterion = build_criterion(self.args).to('cuda')

        def test_train_iteration(dataloader, detr, criterion):
            images, tgt_dict = next(iter(train_dataloader))
            images = images.to('cuda')
            tgt_dict = {k: v.to('cuda') for k, v in tgt_dict.items()}
            torch.stack([v for v in criterion(detr(images), tgt_dict)[0].values()]).sum().backward()

        self.globals_dict['test_train_iteration'] = test_train_iteration
        self.globals_dict['dataloader'] = train_dataloader
        self.globals_dict['detr'] = detr
        self.globals_dict['criterion'] = criterion

        stmt = 'test_train_iteration(dataloader, detr, criterion)'
        timer = Timer(stmt=stmt, globals=self.globals_dict)
        t = timer.timeit(number=5)._median*1e3

        print(f"Memory full model on COCO (train iteration): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time full model on COCO (train iteration): {t: .1f} ms")


class TestModelsForwardOnly(unittest.TestCase):
    def setUp(self):
        print("")

        self.args = get_parser().parse_args()
        self.args.num_classes = 91
        self.globals_dict = {}

        self.pixel_H = 1024
        self.pixel_W = 1024
        self.feat_H = 32
        self.feat_W = 32

        self.args.lr_backbone = 1e-5
        self.args.lr_encoder = 1e-4
        self.args.lr_decoder = 1e-4

    def tearDown(self):
        torch.cuda.reset_peak_memory_stats()

    def test_backbone(self):
        backbone = build_backbone(self.args).to('cuda')

        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W, device='cuda')
        images = nested_tensor_from_image_list(images)

        self.globals_dict['backbone'] = backbone
        self.globals_dict['images'] = images

        timer = Timer(stmt='backbone(images)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory backbone (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time backbone (forward): {t: .1f} ms")

    def test_sine_position(self):
        sine_pos_encoder = SinePositionEncoder()

        features = torch.randn(self.args.batch_size, self.args.feat_dim, self.feat_H, self.feat_W, device='cuda')
        feature_masks = torch.randn(self.args.batch_size, self.feat_H, self.feat_W, device='cuda') > 0

        self.globals_dict['sine_pos_encoder'] = sine_pos_encoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks

        timer = Timer(stmt='sine_pos_encoder(features, feature_masks)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory sine pos (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time sine pose (forward): {t: .1f} ms")

    def test_encoder(self):
        encoder = build_encoder(self.args).to('cuda')

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')
        feature_masks = torch.randn(self.args.batch_size, self.feat_H, self.feat_W, device='cuda') > 0
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')

        self.globals_dict['encoder'] = encoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos_encodings'] = pos_encodings

        timer = Timer(stmt='encoder(features, feature_masks, pos_encodings)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory encoder (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time encoder (forward): {t: .1f} ms")

    def test_global_decoder(self):
        self.args.decoder_type = 'global'
        decoder = build_decoder(self.args).to('cuda')

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')
        feature_masks = torch.randn(self.args.batch_size, self.feat_H, self.feat_W, device='cuda') > 0
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')

        self.globals_dict['decoder'] = decoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos_encodings'] = pos_encodings

        timer = Timer(stmt='decoder(features, feature_masks, pos_encodings)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory global decoder (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time global decoder (forward): {t: .1f} ms")

    def test_sample_decoder(self):
        self.args.decoder_type = 'sample'
        decoder = build_decoder(self.args).to('cuda')

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')
        feature_masks = torch.randn(self.args.batch_size, self.feat_H, self.feat_W, device='cuda') > 0
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')

        self.globals_dict['decoder'] = decoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos_encodings'] = pos_encodings

        timer = Timer(stmt='decoder(features, feature_masks, pos_encodings)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory sample decoder (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time sample decoder (forward): {t: .1f} ms")

    def test_detr(self):
        detr = build_detr(self.args).to('cuda')

        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W, device='cuda')
        images = nested_tensor_from_image_list(images)

        self.globals_dict['detr'] = detr
        self.globals_dict['images'] = images

        timer = Timer(stmt='detr(images)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory DETR model (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time DETR model (forward): {t: .1f} ms")

    def test_matcher(self):
        matcher = build_matcher(self.args).to('cuda')

        num_slots_total = self.args.batch_size * self.args.num_init_slots
        logits = torch.randn(num_slots_total, self.args.num_classes+1, device='cuda')
        boxes = torch.abs(torch.randn(num_slots_total, 4, device='cuda'))
        batch_idx, _ = torch.randint(self.args.batch_size, (num_slots_total,), device='cuda').sort()
        pred_dict = {'logits': logits, 'boxes': boxes, 'batch_idx': batch_idx, 'layer_id': 6, 'iter_id': 1}

        num_target_boxes_total = 20
        labels = torch.randint(self.args.num_classes, (num_target_boxes_total,), device='cuda')
        boxes = torch.abs(torch.randn(num_target_boxes_total, 4, device='cuda'))
        sizes, _ = torch.randint(num_target_boxes_total, (self.args.batch_size+1,), device='cuda').sort()
        tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

        self.globals_dict['matcher'] = matcher
        self.globals_dict['pred_dict'] = pred_dict
        self.globals_dict['tgt_dict'] = tgt_dict

        timer = Timer(stmt='matcher(pred_dict, tgt_dict)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory matcher (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time matcher (forward): {t: .1f} ms")

    def test_criterion(self):
        criterion = build_criterion(self.args).to('cuda')

        num_slots_total = self.args.batch_size * self.args.num_init_slots
        logits = torch.randn(num_slots_total, self.args.num_classes+1, device='cuda', requires_grad=True)
        boxes = torch.abs(torch.randn(num_slots_total, 4, device='cuda', requires_grad=True))
        batch_idx, _ = torch.randint(self.args.batch_size, (num_slots_total,), device='cuda').sort()
        pred_list = [{'logits': logits, 'boxes': boxes, 'batch_idx': batch_idx, 'layer_id': 6, 'iter_id': 1}]

        num_target_boxes_total = 20
        labels = torch.randint(self.args.num_classes, (num_target_boxes_total,), device='cuda')
        boxes = torch.abs(torch.randn(num_target_boxes_total, 4, device='cuda'))
        sizes, _ = torch.randint(num_target_boxes_total, (self.args.batch_size+1,), device='cuda').sort()
        tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

        self.globals_dict['criterion'] = criterion
        self.globals_dict['pred_list'] = pred_list
        self.globals_dict['tgt_dict'] = tgt_dict

        timer = Timer(stmt='criterion(pred_list, tgt_dict)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory criterion (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time criterion (forward): {t: .1f} ms")

    def test_detr_with_criterion(self):
        detr = build_detr(self.args).to('cuda')
        criterion = build_criterion(self.args).to('cuda')

        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W, device='cuda')
        images = nested_tensor_from_image_list(images)

        num_target_boxes_total = 20
        labels = torch.randint(self.args.num_classes, (num_target_boxes_total,), device='cuda')
        boxes = torch.abs(torch.randn(num_target_boxes_total, 4, device='cuda'))
        sizes, _ = torch.randint(num_target_boxes_total, (self.args.batch_size+1,), device='cuda').sort()
        tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

        self.globals_dict['detr'] = detr
        self.globals_dict['criterion'] = criterion
        self.globals_dict['images'] = images
        self.globals_dict['tgt_dict'] = tgt_dict

        timer = Timer(stmt='criterion(detr(images), tgt_dict)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory DETR with criterion (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time DETR with criterion (forward): {t: .1f} ms")


class TestModelsWithBackward(unittest.TestCase):
    def setUp(self):
        print("")

        self.args = get_parser().parse_args()
        self.args.num_classes = 91
        self.globals_dict = {}

        self.pixel_H = 1024
        self.pixel_W = 1024
        self.feat_H = 32
        self.feat_W = 32

        self.args.lr_backbone = 1e-5
        self.args.lr_encoder = 1e-4
        self.args.lr_decoder = 1e-4

    def tearDown(self):
        torch.cuda.reset_peak_memory_stats()

    def test_backbone(self):
        backbone = build_backbone(self.args).to('cuda')

        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W, device='cuda')
        images = nested_tensor_from_image_list(images)

        self.globals_dict['backbone'] = backbone
        self.globals_dict['images'] = images

        timer = Timer(stmt='backbone(images)[-1].tensor.sum().backward()', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory backbone (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time backbone (backward): {t: .1f} ms")

    def test_encoder(self):
        encoder = build_encoder(self.args).to('cuda')

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')
        feature_masks = torch.randn(self.args.batch_size, self.feat_H, self.feat_W, device='cuda') > 0
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')

        self.globals_dict['encoder'] = encoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos'] = pos_encodings

        timer = Timer(stmt='encoder(features, feature_masks, pos).sum().backward()', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory encoder (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time encoder (backward): {t: .1f} ms")

    def test_global_decoder(self):
        self.args.decoder_type = 'global'
        decoder = build_decoder(self.args).to('cuda')

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')
        feature_masks = torch.randn(self.args.batch_size, self.feat_H, self.feat_W, device='cuda') > 0
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')

        self.globals_dict['decoder'] = decoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos'] = pos_encodings

        timer = Timer(stmt='decoder(features, feature_masks, pos)[0].sum().backward()', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory global decoder (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time global decoder (backward): {t: .1f} ms")

    def test_sample_decoder(self):
        self.args.decoder_type = 'sample'
        decoder = build_decoder(self.args).to('cuda')

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')
        feature_masks = torch.randn(self.args.batch_size, self.feat_H, self.feat_W, device='cuda') > 0
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')

        self.globals_dict['decoder'] = decoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos'] = pos_encodings

        timer = Timer(stmt='decoder(features, feature_masks, pos)[0].sum().backward()', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory sample decoder (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time sample decoder (backward): {t: .1f} ms")

    def test_detr(self):
        detr = build_detr(self.args).to('cuda')

        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W, device='cuda')
        images = nested_tensor_from_image_list(images)

        self.globals_dict['detr'] = detr
        self.globals_dict['images'] = images
        self.globals_dict['keys'] = ['logits', 'boxes']

        stmt = 'torch.cat([v for k, v in detr(images)[0].items() if k in keys], dim=1).sum().backward()'
        timer = Timer(stmt=stmt, globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory DETR model (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time DETR model (backward): {t: .1f} ms")

    def test_criterion(self):
        criterion = build_criterion(self.args).to('cuda')

        def generate_pred_list(self):
            num_slots_total = self.args.batch_size * self.args.num_init_slots
            logits = torch.randn(num_slots_total, self.args.num_classes+1, device='cuda', requires_grad=True)
            boxes = torch.abs(torch.randn(num_slots_total, 4, device='cuda', requires_grad=True))
            batch_idx, _ = torch.randint(self.args.batch_size, (num_slots_total,), device='cuda').sort()
            pred_list = [{'logits': logits, 'boxes': boxes, 'batch_idx': batch_idx, 'layer_id': 6, 'iter_id': 1}]

            return pred_list

        num_target_boxes_total = 20
        labels = torch.randint(self.args.num_classes, (num_target_boxes_total,), device='cuda')
        boxes = torch.abs(torch.randn(num_target_boxes_total, 4, device='cuda'))
        sizes, _ = torch.randint(num_target_boxes_total, (self.args.batch_size+1,), device='cuda').sort()
        tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

        self.globals_dict['criterion'] = criterion
        self.globals_dict['generate_pred_list'] = generate_pred_list
        self.globals_dict['self'] = self
        self.globals_dict['tgt_dict'] = tgt_dict

        stmt = 'torch.stack([v for v in criterion(generate_pred_list(self), tgt_dict)[0].values()]).sum().backward()'
        timer = Timer(stmt=stmt, globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory criterion (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time criterion (backward): {t: .1f} ms")

    def test_detr_with_criterion(self):
        detr = build_detr(self.args).to('cuda')
        criterion = build_criterion(self.args).to('cuda')

        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W, device='cuda')
        images = nested_tensor_from_image_list(images)

        num_target_boxes_total = 20
        labels = torch.randint(self.args.num_classes, (num_target_boxes_total,), device='cuda')
        boxes = torch.abs(torch.randn(num_target_boxes_total, 4, device='cuda'))
        sizes, _ = torch.randint(num_target_boxes_total, (self.args.batch_size+1,), device='cuda').sort()
        tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

        self.globals_dict['detr'] = detr
        self.globals_dict['criterion'] = criterion
        self.globals_dict['images'] = images
        self.globals_dict['tgt_dict'] = tgt_dict

        stmt = 'torch.stack([v for v in criterion(detr(images), tgt_dict)[0].values()]).sum().backward()'
        timer = Timer(stmt=stmt, globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory DETR with criterion (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time DETR with criterion (backward): {t: .1f} ms")


class TestLoadingFromOriginalDETR(unittest.TestCase):

    def setUp(self):
        self.args = get_parser().parse_args()
        self.args.num_classes = 91

        original_detr_url = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
        self.state_dict = torch.hub.load_state_dict_from_url(original_detr_url)['model']

    def test_backbone(self):
        backbone = build_backbone(self.args)
        backbone.load_from_original_detr(self.state_dict)

    def test_encoder(self):
        encoder = build_encoder(self.args)
        encoder.load_from_original_detr(self.state_dict)

    def test_global_decoder(self):
        self.args.decoder_type = 'global'
        decoder = build_decoder(self.args)
        decoder.load_from_original_detr(self.state_dict)


if __name__ == '__main__':
    unittest.main()
