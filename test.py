import yaml, argparse, os, json, copy, multiprocessing, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataloader.nusc_loader import NuScenesloader
from tracking.nusc_tracker import Tracker
from nuscenes.nuscenes import NuScenes
from data.script.NUSC_CONSTANT import *
from utils.io import load_file, dict_to_yaml
from utils.script import tra_trans_det
from itertools import product
from typing import List
from tqdm import tqdm

parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--process', type=int, default=1)
# paths
parser.add_argument('--nusc_path', type=str, default='/data1/wyt_dataset1/nuscenes/')
parser.add_argument('--config_path', type=str, default='config/nusc_config.yaml')
parser.add_argument('--detection_3d_path', type=str, default='data/detector/val/nuscenes_val_centerpoint_3d.json')
parser.add_argument('--detection_2d_path', type=str, default='data/utils/cascade_rcnn_4hz/nuscenes_val_cascade_2d_4hz.json')
parser.add_argument('--first_token_path', type=str, default='data/utils/first_token_table/trainval/nuscenes_val_first_sample_tokens.json')
parser.add_argument('--result_path', type=str, default='Fusion_Poly_EXP/result/single_stage/')
parser.add_argument('--eval_path', type=str, default='Fusion_Poly_EXP/eval_results/single_stage/')
# auto finetune
parser.add_argument('--auto_finetune', type=bool, default=False)
parser.add_argument('--process_num', type=int, default=5)
parser.add_argument('--auto_finetune_config_path', type=str, default='config/auto_finetune.yaml')
parser.add_argument('--auto_save_path', type=str, default='/data1/wyt_dataset1/Fusion_Poly_EXP/fusion_poly_auto_finetune/IoU_ndm')
parser.add_argument('--auto_result_path', type=str, default='final_result')
parser.add_argument('--auto_eval_path', type=str, default='final_result')
args = parser.parse_args()


def main(result_path, token, process, nusc_loader):
    # PolyMOT modal is completely dependent on the detector modal
    result = {
        "results": {},
        "meta": {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
    }
    # tracking and output file
    nusc_tracker = Tracker(config=nusc_loader.config, nusc=nusc_loader.nusc)
    for frame_data in tqdm(nusc_loader, desc='Running', total=len(nusc_loader) // process, position=token):  
        if process > 1 and frame_data['seq_id'] % process != token:
            continue
        sample_token = frame_data['sample_token']

        # if don't use high frequency data, skip this frame and don't tracking
        if len(sample_token.split('_')) > 1 and nusc_loader.config['basic']['freq'] == 'low': continue
        # track each sequence
        nusc_tracker.tracking(frame_data)
        # skip non-key-frame, don't output any boxes
        if not frame_data['is_key_frame']: continue
        # output process
        sample_results = []
        if 'no_val_track_result' not in frame_data:
            for predict_box in frame_data['box_track_res']:
                box_result = {
                    "sample_token": sample_token,
                    "translation": [float(predict_box.center[0]), float(predict_box.center[1]),
                                    float(predict_box.center[2])],
                    "size": [float(predict_box.wlh[0]), float(predict_box.wlh[1]), float(predict_box.wlh[2])],
                    "rotation": [float(predict_box.orientation[0]), float(predict_box.orientation[1]),
                                 float(predict_box.orientation[2]), float(predict_box.orientation[3])],
                    "velocity": [float(predict_box.velocity[0]), float(predict_box.velocity[1])],
                    "tracking_id": str(predict_box.tracking_id),
                    "tracking_name": predict_box.name,
                    "tracking_score": predict_box.score,
                }
                sample_results.append(box_result.copy())

        # add to the output file
        if sample_token in result["results"]:
            result["results"][sample_token] = result["results"][sample_token] + sample_results
        else:
            result["results"][sample_token] = sample_results

    # sort track result by the tracking score
    for sample_token in result["results"].keys():
        confs = sorted(
            [
                (-d["tracking_score"], ind)
                for ind, d in enumerate(result["results"][sample_token])
            ]
        )
        result["results"][sample_token] = [
            result["results"][sample_token][ind]
            for _, ind in confs[: min(500, len(confs))]
        ]

    # write file
    if process > 1:
        json.dump(result, open(result_path + str(token) + ".json", "w"))
    else:
        json.dump(result, open(result_path + "/results.json", "w"))


def eval(result_path, eval_path, nusc_path):
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory as track_configs
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=result_path,
        eval_set="val",
        output_dir=eval_path,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=nusc_path,
    )
    print("result in " + result_path)
    metrics_summary = nusc_eval.main()


def eval_dets(result_path, eval_path, nusc_path):
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.common.config import config_factory as detect_configs
    cfg = detect_configs('detection_cvpr_2019')
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_path, verbose=True)
    nusc_eval = DetectionEval( nusc,
        config=cfg,
        result_path=result_path,
        eval_set="val",
        output_dir=eval_path,
        verbose=True,
    )
    print("result in " + result_path)
    metrics_summary = nusc_eval.main()


def tra_eval_det(result_path, eval_path, nusc_path):
    '''
    Convert trajectories to detections format and evaluate
    '''
    # ensure the path exists
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)

    # transfer tracking result to detection result and evaluate
    tra_trans_det(nusc_tracking_path=result_path, nusc_det_path=result_path)
    eval_dets(os.path.join(result_path, 'results_det.json'), eval_path, nusc_path)


def find_best_para(sort_metric, results):
    try:
        sample_para = results[0][0]
    except:
        print("No linear_search_parameters result!!! Please check the result file!!!")
        raise ValueError
    
    select_func = max if sort_metric in INCREASING_EVAL_METRIC else min
    if isinstance(sample_para, dict):
        best_para_dict = {}
        for _, configs in results:
            for category, info in configs.items():
                if category == "all":
                    continue
                if category not in best_para_dict or select_func(
                        [info[sort_metric], best_para_dict[category][sort_metric]]) == info[sort_metric]:
                    best_para_dict[category] = info
        all_metric_avg = sum(info[sort_metric] for info in best_para_dict.values()) / len(best_para_dict)

        best_para = {info['category_idx']: info['config'] for category, info in best_para_dict.items() if
                     category != "all"}
        best_configs = {category: info for category, info in best_para_dict.items()}
        best_configs['all'] = {sort_metric: all_metric_avg}
    else:
        best_idx = select_func(range(len(results)), key=lambda i: results[i][1]['all'][sort_metric])
        best_para, best_configs = results[best_idx]

    return best_para, best_configs


def search_parameter(interval, parameter, config_dict_tree, result_path, metrics_form, module):
    all_configs, all_results, best_configs = {}, {}, {}
    # iterative experiments
    for iter_num, iter in enumerate(interval * 100):
        iter = round(iter)
        paras = iter / 100
        eval_path = f'{result_path}/{parameter}/eval_results/eval{iter}/'
        res_path = f'{result_path}/{parameter}/results/result{iter}/'
        metrics_summary_path = os.path.join(eval_path, 'metrics_summary.json')

        # replace specific config with progression value
        search_cfg = config_dict_tree.set_value(key=parameter, s_value=paras, module=module)
        all_configs[iter] = copy.deepcopy(search_cfg)

        # inference Offline-Poly with changed config, and save result
        if not os.path.exists(metrics_summary_path):
            run_nusc_polymot(config_dict_tree.config, res_path, eval_path)

        # record each epoch eval result
        if metrics_form == 'tracking_metrics':
            all_results[iter] = load_file(metrics_summary_path)[metrics_form]['label_metrics'][sort_metric]
        elif metrics_form == 'detection_metrics':
            all_results[iter], metrics_dict = {}, {}
            original_dict = load_file(metrics_summary_path)[metrics_form]['label_tp_errors']
            for cls_name, errors in original_dict.items():
                if cls_name in CLASS_SEG_TO_STR_CLASS:
                    metrics_dict[cls_name] = errors[sort_metric]
            all_results[iter] = metrics_dict
        else:
            raise NotImplementedError

        # Selection of select_func according to sort_metric
        select_func = max if sort_metric in INCREASING_EVAL_METRIC else min

        # sort accuracy and save parameters
        if isinstance(search_cfg, dict):
            for cls_label, cls_name in CLASS_STR_TO_SEG_CLASS.items():
                # list all configs and eval results
                cls_cfg = [cfg[cls_label] for _, cfg in all_configs.items()]
                cls_acc = [result[cls_name] for _, result in all_results.items()]

                # best config and amota
                cls_cfg = {
                    'category': cls_name,
                    'category_idx': cls_label,
                    sort_metric: select_func(cls_acc),
                    'exp_idx': cls_acc.index(select_func(cls_acc)),
                    'config': cls_cfg[cls_acc.index(select_func(cls_acc))],
                }
                best_configs[cls_name] = cls_cfg

            best_configs['all'] = {
                sort_metric: sum([best_configs[cls][sort_metric] for cls in CLASS_SEG_TO_STR_CLASS]) / 7
            }

            best_para = {cls_label: best_configs[cls_name]['config'] for cls_name, cls_label in
                         CLASS_SEG_TO_STR_CLASS.items()}
        else:
            all_results[iter] = load_file(metrics_summary_path)[metrics_form][sort_metric]
            iter_results = [_ for i, _ in all_results.items()]
            iter_cfg = [_ for i, _ in all_configs.items()]
            best_configs['all'] = {
                sort_metric: select_func(iter_results),
                'exp_idx': iter_results.index(select_func(iter_results)),
                'config': iter_cfg[iter_results.index(select_func(iter_results))],
            }
            best_para = best_configs['all']['config']

    return best_para, best_configs


def multi_threading_search(interval, step, process_num, parameter, config_dict_tree, result_path,
                           metrics_form, module):
    ### attention: config is deleted in the process of searching
    start, end = interval
    process_num = min(process_num, len(np.arange(start, end, step)))
    divided_interval = np.array_split(np.arange(start, end, step), process_num)
    results = []

    if process_num > 1:
        with ProcessPoolExecutor(max_workers=process_num) as executor:
            futures = {
                executor.submit(search_parameter, process_interval, parameter, config_dict_tree, result_path,
                                metrics_form, module):
                    (process_interval, parameter, config_dict_tree, result_path, metrics_form, module) for
                process_interval in divided_interval
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    results.append(future.result())
                except:
                    print(traceback.format_exc())
    else:
        for process_interval in divided_interval:
            results.append(search_parameter(process_interval, parameter, config_dict_tree, result_path, metrics_form, module))

    return results


def linear_search_parameters(online_cfg, intervals: List[List], steps: List[float], parameters: List[str], epoch: int,
                             modules: List[str], sort_metric: str = 'amota', process_num: int = 1, traversal: bool = False):
    """
    linear search specific parameters based on the sort metric in the nuScenes val set.
    :param online_cfg: dict, raw config for online tracking
    :param intervals: List[List], range of to be fine-tuned parameters, [left, right]
    :param steps: List[float], linear search step size
    :param parameters: List[str], the name of parameter
    :param modules: List[str], the tracking module of the parameter
    :param epoch: int, the epoch idx of the current auto finetune
    :param sort_metric: str, the metric used to sort, default 'amota' in the nuscenes dataset
    :return: parameter under the best performance
    """
    # load raw config
    from auto_finetune_module.utils.config_dict_tree import ConfigDictTree

    online_config = copy.deepcopy(online_cfg)

    # check the sort metric
    if sort_metric in TRACKING_METRIC:
        metrics_form = 'tracking_metrics'
    elif sort_metric in DETECTION_METRIC:
        metrics_form = 'detection_metrics'
    else:
        raise ValueError(f'Invalid sort metric: {sort_metric}')
    
    config_dict_tree = ConfigDictTree(online_config)

    # multiple parameters fine-tuning
    if not traversal:
        for interval, step, parameter, module in zip(intervals, steps, parameters, modules):
            assert interval[0] <= interval[1], "invalid range."

            root_path = args.auto_save_path
            result_path = os.path.join(root_path, f'epoch{epoch}')
            best_cfg_path = os.path.join(root_path, f'epoch{epoch}', f'best_parameter_{parameter}_epoch_{epoch}.yaml')
            
            results = multi_threading_search(interval=interval,
                                             step=step,
                                             process_num=process_num,
                                             parameter=parameter,
                                             config_dict_tree=config_dict_tree,
                                             result_path=result_path,
                                             metrics_form=metrics_form,
                                             module=module
                                             )
            
            # find the best Para from multi_threading_search results
            best_para, best_configs = find_best_para(sort_metric, results)
            _ = config_dict_tree.set_value(key=parameter, s_value=best_para, module=module)

            # write config under best performance
            dict_to_yaml(best_configs, best_cfg_path)
            print(f'writing best configs for {parameter} under {epoch} epoch in folder: ' +
                  os.path.abspath(best_cfg_path))

        return config_dict_tree.config
    
    else: 
        # Traversal Mode: Obtain the config tree under all parameter combinations
        config_trees = get_traversal_config_trees(config_dict_tree, intervals,
                                                  steps, parameters, modules)
        result_path = os.path.join(args.auto_save_path)
        best_cfg_path = os.path.join(result_path, f'best_parameter_traversal.yaml')

        if process_num <= 1:
            all_results = []
            # iterative experiments
            for iter in range(len(config_trees)):
                single_dump_res = single_dump(
                    (iter, config_trees[iter], result_path, metrics_form, sort_metric))
                # record the accuracy of each experiment
                all_results.append(single_dump_res)
        else: 
            # multi-threading experiments
            num_workers = min(len(config_trees), process_num)
            # Prepare arguments list
            args_list = [
                (i, config_tree, result_path, metrics_form, sort_metric)
                for i, config_tree in enumerate(config_trees)
            ]
            # Create process pool
            with multiprocessing.Pool(num_workers) as pool:
                # Use tqdm for progress tracking
                all_results = list(tqdm(
                    pool.imap(single_dump, args_list),
                    total=len(args_list),
                    desc="Running experiments"
                ))

        # Selection of select_func according to sort_metric
        select_func = max if sort_metric in INCREASING_EVAL_METRIC else min
        best_config_tree = config_trees[all_results.index(select_func(all_results))]

        # write config under best performance
        dict_to_yaml(best_config_tree.config, best_cfg_path)
        print(f'writing best configs under traversal mode in folder: ' +
              os.path.abspath(best_cfg_path))
        
        return best_config_tree.config


def single_dump(arg_list):
    iter, config_tree, result_path, metrics_form, sort_metric = arg_list
    eval_path = f'{result_path}/eval_results/eval{iter}/'
    res_path = f'{result_path}/results/result{iter}/'
    metrics_summary_path = os.path.join(eval_path, 'metrics_summary.json')
    results_file_path = os.path.join(res_path, 'results.json')

    # inference Offline-Poly with changed config, and save result
    if not os.path.exists(metrics_summary_path):
        run_nusc_polymot(config_tree.config, res_path, eval_path,
                             only_eval=True if os.path.exists(results_file_path) else False)

    # record the accuracy of each experiment
    return load_file(metrics_summary_path)[metrics_form][sort_metric]


def get_traversal_config_trees(config_dict_tree, intervals, steps, parameters, modules):
    """
    :param config_dict_tree: the original config tree
    :param intervals: List[List], range of to be fine-tuned parameters, [left, right]
    :param steps: List[float], linear search step size
    :param parameters: List[str], the name of parameter
    :param modules: List[str], the tracking module of the parameter
    :return: list of dicts, traversal of all parameter combinations with module and submodule info
    """
    # For each parameter, generate the list of values based on its interval and step size
    param_values = []
    for i, param in enumerate(parameters):
        left, right = intervals[i]
        step = steps[i]

        # Generate values within the range [left, right] with the given step size
        values = np.arange(left, right + step, step).tolist()
        param_values.append(values)

    # Create all combinations of parameters using itertools.product
    all_combinations = product(*param_values)

    # Build the result list with dictionaries containing parameter, module, and submodule info
    config_trees = []
    for combination in all_combinations:
        config_tree = copy.deepcopy(config_dict_tree)

        # set the config of each sub config_tree
        for i, param in enumerate(combination):
            _ = config_tree.set_value(key=parameters[i], s_value=param, module=modules[i])

        config_trees.append(config_tree)

    return config_trees


def run_nusc_polymot(config, result_path, eval_path, only_eval=False):
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)

    if config['basic']['freq'] == 'high': config['basic']['LiDAR_interval'] = 0.25
    # save config file
    online_config_save_path = os.path.join(eval_path, 'config.yaml')
    dict_to_yaml(config, online_config_save_path)
    print('writing config in folder: ' + os.path.abspath(eval_path))
    # load dataloader
    nusc_loader = NuScenesloader(args.detection_3d_path, 
                                 args.detection_2d_path,
                                 args.first_token_path,
                                 args.nusc_path,
                                 config)
    print('writing result in folder: ' + os.path.abspath(result_path))

    if not only_eval:
        if args.process > 1:
            result_temp_path = result_path
            os.makedirs(result_temp_path, exist_ok=True)
            pool = multiprocessing.Pool(args.process)
            for token in range(args.process):
                pool.apply_async(main, args=(result_temp_path, token, args.process, nusc_loader))
            pool.close()
            pool.join()
            results = {'results': {}, 'meta': {}}
            # combine the results of each process
            try:
                for token in range(args.process):
                    result = json.load(open(os.path.join(result_temp_path, str(token) + '.json'), 'r'))
                    results["results"].update(result["results"])
                    results["meta"].update(result["meta"])
            except:
                print(traceback.format_exc())
            json.dump(results, open(result_path + '/results.json', "w"))
        else:
            # single process inference
            main(result_path, 0, 1, nusc_loader)
        print('result is written in folder: ' + os.path.abspath(result_path))

    # eval result
    eval(os.path.join(result_path, 'results.json'), os.path.join(eval_path, 'tracking/'), args.nusc_path)
    tra_eval_det(result_path, os.path.join(eval_path, 'detection/'), args.nusc_path)
    summary_dict = {
        'tracking_metrics': load_file(os.path.join(eval_path, 'tracking/metrics_summary.json')),
        'detection_metrics': load_file(os.path.join(eval_path, 'detection/metrics_summary.json'))
    }
    json.dump(summary_dict, open(eval_path + "/metrics_summary.json", "w"), indent=2)


if __name__ == "__main__":
    if not args.auto_finetune:
        # single inference, load and save config
        config = yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader)
        
        # run Offline-Poly
        run_nusc_polymot(config, args.result_path, args.eval_path)
    else:
        # auto-finetune implementation
        auto_config = yaml.load(open(args.auto_finetune_config_path, 'r'), Loader=yaml.Loader)
        online_config = copy.deepcopy(yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader))

        auto_root_path = args.auto_save_path
        os.makedirs(auto_root_path, exist_ok=True)

        auto_config_save_path = os.path.join(auto_root_path, f'auto_finetune.yaml')
        dict_to_yaml(auto_config, auto_config_save_path)

        # auto-finetune basic setting
        num_epochs = auto_config['basic']['num_epochs']
        sort_metric = auto_config['basic']['sort_metric']
        traversal = auto_config['basic']['traversal']
        process_num = args.process_num

        # auto-finetune parameter setting
        paras, modules = auto_config['paras']['names'], auto_config['paras']['module_names']
        intervals, steps = auto_config['paras']['intervals'], auto_config['paras']['steps']

        num_paras = len(paras)
        assert num_paras == len(modules) == len(intervals) == len(steps)
        
        # auto-finetune experiments counts for each paras
        counts_each_epoch = np.array([_[1] - _[0] for _ in intervals]) // np.array(steps)
        
        if not traversal:
            for counts, para_name in zip(counts_each_epoch, paras):
                print(f'[AUTO FINETUNE - Epoch Mode] Need {int(counts):>3} experiments for {para_name} in one epoch.')
            print(f'[AUTO FINETUNE - Epoch Mode] Total {int(np.sum(counts_each_epoch))} experiments in one epoch.\n')
            for epoch in range(num_epochs):
                # linear search parameters
                print(f'[AUTO FINETUNE - Epoch Mode] Epoch {epoch + 1}/{num_epochs}, '
                    f'Total {num_paras} parameters are searching for the optimal value.')
                online_config = linear_search_parameters(online_cfg=online_config,
                                                         intervals=intervals, 
                                                         steps=steps, 
                                                         parameters=paras, 
                                                         modules=modules,
                                                         epoch=epoch, 
                                                         sort_metric=sort_metric,
                                                         process_num=process_num,
                                                         traversal=traversal)

                # saving the optimal cfg of the current epoch
                online_config_save_path = os.path.join(auto_root_path, f'best_online_config_epoch_{epoch}.yaml')
                dict_to_yaml(online_config, online_config_save_path)

                # eval the final auto-finetune result
                auto_result_path = os.path.join(args.auto_save_path, args.auto_result_path, f'epoch_{epoch}')
                auto_eval_path = os.path.join(args.auto_save_path, args.auto_eval_path, f'epoch_{epoch}')
                auto_metrics_summary_path = os.path.join(auto_eval_path, 'metrics_summary.json')
                if not os.path.exists(auto_metrics_summary_path):
                    run_nusc_polymot(online_config, auto_result_path, auto_eval_path)
        else: # traversal mode
            for counts, para_name in zip(counts_each_epoch, paras):
                print(f'[AUTO FINETUNE - Traversal Mode] Need {int(counts):>3} experiments for {para_name}.')
            print(
                f'[AUTO FINETUNE - Traversal Mode] Total {int(np.prod(counts_each_epoch))} experiments for auto-finetune.\n')
            online_config = linear_search_parameters(online_cfg=online_config,
                                                    intervals=intervals, 
                                                    steps=steps, 
                                                    parameters=paras, 
                                                    modules=modules,
                                                    epoch=None, 
                                                    sort_metric=sort_metric,
                                                    process_num=process_num,
                                                    traversal=traversal)

            # saving the optimal cfg of the current epoch
            online_config_save_path = os.path.join(auto_root_path, f'best_online_config_traversal.yaml')
            dict_to_yaml(online_config, online_config_save_path)
            
            # eval the final auto-finetune result
            auto_result_path = os.path.join(args.auto_save_path, args.auto_result_path, f'traversal')
            auto_eval_path = os.path.join(args.auto_save_path, args.auto_eval_path, f'traversal')
            auto_metrics_summary_path = os.path.join(auto_eval_path, 'metrics_summary.json')
            if not os.path.exists(auto_metrics_summary_path):
                run_nusc_polymot(online_config, auto_result_path, auto_eval_path)


