import argparse
from model.utils.config import cfg, cfg_from_file, cfg_from_list


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='source training dataset',
                        default='pascal_voc_0712', type=str)
    parser.add_argument('--dataset_t', dest='dataset_t',
                        help='target training dataset',
                        default='clipart', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101 res50',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--gamma', dest='gamma',
                        help='value of gamma',
                        default=5, type=float)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--load_name', dest='load_name',
                        help='path to load models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')

    parser.add_argument('--detach', dest='detach',
                        help='whether use detach',
                        action='store_false')
    parser.add_argument('--ef', dest='ef',
                        help='whether use exponential focal loss',
                        action='store_true')
    parser.add_argument('--lc', dest='lc',
                        help='whether use context vector for pixel level',
                        action='store_true')
    parser.add_argument('--gc', dest='gc',
                        help='whether use context vector for global level',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--eta', dest='eta',
                        help='trade-off parameter between detection loss and domain-alignment loss. Used for Car datasets',
                        default=0.1, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--test_cache', dest='test_cache',
                        action='store_true')
    args = parser.parse_args()
    return args


def set_dataset_args(args, test=False):
    if not test:
        data2imdb_dict = get_data2imdb_dict()
        if args.dataset in data2imdb_dict:
            args.imdb_name = data2imdb_dict[args.dataset]
        elif args.dataset == "pascal_voc":
            args.imdb_name = "voc_2007_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_water":
            args.imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
            args.imdbval_name = "voc_clipart_2007_trainval+voc_clipart_2012_trainval"
            args.imdb_name_cycle = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_cycleclipart":
            args.imdb_name = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.imdbval_name = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_cyclewater":
            args.imdb_name = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.imdbval_name = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_0712":
            args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            args.imdbval_name = "voc_2007_test"
            args.imdb_name_cycle = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "foggy_cityscape":
            args.imdb_name = "foggy_cityscape_trainval"
            args.imdbval_name = "foggy_cityscape_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '30']
        elif args.dataset == "vg":
            args.imdb_name = "vg_150-50-50_minitrain"
            args.imdbval_name = "vg_150-50-50_minival"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "cityscape":
            args.imdb_name = "cityscape_trainval"
            args.imdbval_name = "cityscape_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '30']
        elif args.dataset == "sim10k":
            args.imdb_name = "sim10k_train"
            args.imdbval_name = "sim10k_train"
            # "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.imdb_name_cycle = "sim10k_cycle_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "sim10k_cycle":
            args.imdb_name = "sim10k_cycle_train"
            args.imdbval_name = "sim10k_cycle_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        # cityscape dataset for only car classes.
        # elif args.dataset == "cityscape_kitti":
        #     args.imdb_name = "cityscape_kitti_trainval"
        #     args.imdbval_name = "cityscape_kitti_trainval"
        #     args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
        #                      '30']
        data2imdb_inner_dict = get_data2imdb_inner_dict()
        if args.dataset_t in data2imdb_inner_dict:
            args.imdb_name_target = data2imdb_inner_dict[args.dataset_t]
        elif args.dataset_t == "water":
            args.imdb_name_target = "water_train"
            args.imdbval_name_target = "water_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']
        elif args.dataset_t == "clipart":
            args.imdb_name_target = "clipart_trainval"
            args.imdbval_name_target = "clipart_test"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']
        elif args.dataset_t == "cityscape":
            args.imdb_name_target = "cityscape_trainval"
            args.imdbval_name_target = "cityscape_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '30']
        # cityscape dataset for only car classes.
        elif args.dataset_t == "cityscape_car":
            args.imdb_name_target = "cityscape_car_trainval"
            args.imdbval_name_target = "cityscape_car_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']
        # elif args.dataset_t == "kitti":
        #     args.imdb_name_target = "kitti_trainval"
        #     args.imdbval_name_target = "kitti_trainval"
        #     args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
        #                             '20']
        elif args.dataset_t == "foggy_cityscape":
            args.imdb_name_target = "foggy_cityscape_trainval"
            args.imdbval_name_target = "foggy_cityscape_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '30']
    else:

        if args.dataset == "pascal_voc":
            args.imdb_name = "voc_2007_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES',
                             '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        elif args.dataset == "pascal_voc_0712":
            args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES',
                             '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        elif args.dataset == "sim10k":
            args.imdb_name = "sim10k_val"
            args.imdbval_name = "sim10k_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '30']
        elif args.dataset == "cityscape":
            args.imdb_name = "cityscape_val"
            args.imdbval_name = "cityscape_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "foggy_cityscape":
            args.imdb_name = "foggy_cityscape_test"
            args.imdbval_name = "foggy_cityscape_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "cityscape_kitti":
            args.imdb_name = "cityscape_kitti_val"
            args.imdbval_name = "cityscape_kitti_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "water":
            args.imdb_name = "water_test"
            args.imdbval_name = "water_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '20']
        elif args.dataset == "clipart":
            args.imdb_name = "clipart_trainval"
            args.imdbval_name = "clipart_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '20']
        elif args.dataset == "cityscape_car":
            args.imdb_name = "cityscape_car_val"
            args.imdbval_name = "cityscape_car_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '20']
    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    return args


def get_data2imdbval_dict(imgset, category_imgset='trian'):
    # create canttens

    assert imgset in ['val', 'test', 'train']

    # set canteens
    collected_cts = ["Arts", "Science", "YIH",
                     "UTown", "TechChicken", "TechMixedVeg", "EconomicBeeHoon"]
    excl_cts = ["excl"+x for x in collected_cts]
    all_canteens = collected_cts + excl_cts + ['All']

    # basic setting
    # create dict{ dataset -> imdb_name }
    # dataset format : food{canteen}
    # imdb format : food_{canteen}_{imdgeset}_{category_cateen}_train_mt{}
    #                    [      imageset    ]  [         category        ]

    # 1. create dataset -> dataset_val

    data2imdbval_dict = {}

    for ct in all_canteens:
        dataset = "food{}".format(ct)
        imdbval_name = "food_{}_{}_{}_train".format(
            ct, imgset, ct)
        data2imdbval_dict[dataset] = imdbval_name

    for ct in all_canteens:
        for mtN in [0, 10]:
            if mtN == 0:
                ct_sp = imgset
            else:
                ct_sp = "{}mt{}".format(imgset, mtN)

            # datasets here only support mtN format
            dataset = "food{}mt{}".format(ct, mtN)
            imdbval_name = "food_{}_{}_{}_train_mt{}".format(
                ct, ct_sp, ct, mtN)
            data2imdbval_dict[dataset] = imdbval_name

    # 2. create excl_dataset -> dataset
    # cross domain setting

    for ct in collected_cts:
        for mtN in [0, 10]:
            for fewN in [0, 1, 5]:
                if mtN == 0:
                    mtN_str = ""
                else:
                    mtN_str = "mt{}".format(mtN)

                if fewN == 0:
                    fewN_str = ""
                else:
                    fewN_str = "few{}".format(fewN)

                # datasets here only support mtN format
                # for example:  dataset    -> foodexclArts[mt10]_testArts[few1]
                #               imdb_train -> food_excl
                #               imdb_val   -> food_Arts_inner{}{}val

                dataset = "foodexcl{}{}_test{}{}".format(
                    ct, mtN_str, ct,  fewN_str)

                if fewN == 0:
                    imdbval_name = "food_{}_inner{}{}_excl{}_train_mt{}".format(
                        ct, mtN_str, imgset, ct, mtN)  # innermt10val or innermt10test
                else:
                    imdbval_name = "food_{}_inner{}{}val_excl{}_train_mt{}".format(
                        ct, fewN_str, mtN_str, ct, mtN)  # it not working anymore . it is like innerfew1mt10val: TODO spliting to val and test

                data2imdbval_dict[dataset] = imdbval_name

    # 3. create exclcanteen_finecanteenfewN -> canteenfewN

    for ct in collected_cts:
        for mtN in [10]:
            for fewN in [1, 5, 10]:
                dataset = "foodexcl{}mt{}_fine{}few{}_test{}few{}".format(
                    ct, mtN, ct, fewN, ct, fewN)
                imdbval_name = "food_{}_innerfew{}mt{}val_excl{}_train_mt{}".format(
                    ct, fewN, mtN, ct, mtN)
                data2imdbval_dict[dataset] = imdbval_name

    # 4. extra

    data2imdbval_dict['schoollunch'] = 'schoollunch_{}'.format(args.imgset)
    return data2imdbval_dict

def get_data2imdb_inner_dict(split='innermt10val', category_split='train'):
    # create canttens
    collected_cts = ["Arts", "Science", "YIH",
                     "UTown", "TechChicken", "TechMixedVeg", "EconomicBeeHoon"]
    excl_cts = ["excl"+x for x in collected_cts]
    all_canteens = collected_cts + excl_cts + ['All']

    # create dict{ dataset -> imdb_name }
    data2imdb_dict = {}

    # 1. train on origin mt
    for ct in all_canteens:
        for mtN in [0, 10]:
            if mtN == 0:
                mtNstr = ""
            else:
                mtNstr = "mt{}".format(mtN)
            ct_sp = "{}{}".format(split, mtNstr)

            if mtN == 0:
                imdb_name = "food_{}_{}_{}_{}".format(ct, ct_sp, ct, category_split)
            else:
                imdb_name = "food_{}_{}_{}_{}_mt{}".format(
                    ct, ct_sp, ct, split, mtN)
            dataset = "food{}{}".format(ct, mtNstr)
            data2imdb_dict[dataset] = imdb_name
    return data2imdb_dict

def get_data2imdb_dict(split='train', category_split='train'):
    # create canttens
    collected_cts = ["Arts", "Science", "YIH",
                     "UTown", "TechChicken", "TechMixedVeg", "EconomicBeeHoon"]
    excl_cts = ["excl"+x for x in collected_cts]
    all_canteens = collected_cts + excl_cts + ['All']

    # create dict{ dataset -> imdb_name }
    data2imdb_dict = {}

    # 1. train on origin mt
    for ct in all_canteens:
        for mtN in [0, 10]:
            if mtN == 0:
                mtNstr = ""
            else:
                mtNstr = "mt{}".format(mtN)
            ct_sp = "{}{}".format(split, mtNstr)

            if mtN == 0:
                imdb_name = "food_{}_{}_{}_{}".format(ct, ct_sp, ct, split)
            else:
                imdb_name = "food_{}_{}_{}_{}_mt{}".format(
                    ct, ct_sp, ct, split, mtN)
            dataset = "food{}{}".format(ct, mtNstr)
            data2imdb_dict[dataset] = imdb_name

    # 2. trian on fine
    for ct in collected_cts:
        for mtN in [10]:
            for fewN in [1, 5, 10]:
                dataset = "foodexcl{}mt{}_fine{}few{}".format(
                    ct, mtN, ct, fewN)
                imdb_name = "food_{}_innerfew{}mt{}{}_excl{}_{}_mt{}".format(
                    ct, fewN, mtN, split,  ct, split,  mtN)
                data2imdb_dict[dataset] = imdb_name

    for ct in collected_cts:
        dataset = "food_meta_{}_train".format(ct)
        imdb_name = dataset
        data2imdb_dict[dataset] = imdb_name

    # 4. extra
    data2imdb_dict['schoollunch'] = 'schoollunch_train'
    return data2imdb_dict

    # 5.voc
    data2imdb_dict[''] = 'voc'

def set_food_imdb_name(args):
    data2imdb_dict = get_data2imdb_dict()
    data2imdb_inner_dict = get_data2imdb_inner_dict(split='innermt10val')
    args.imdb_name = data2imdb_dict[args.dataset]
    args.imdb_name_target = data2imdb_inner_dict[args.dataset_t]
    return args
