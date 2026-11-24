"""`python -m radev <command>` —— 发表臂入口（训练/评估 + RADAR 特征缓存）。

  train        决策原生训练 / 评估 / 地板（透传给 radev.train.loop）
  radar-cache  冻结编码器逐 study 特征缓存（plan / run，透传给 radev.train.cache）
"""
import argparse


def cmd_radar_cache(args):
    """参数原样透传给 `radev.train.cache`（旋钮真值在 cache.main 的 parser）。"""
    from .train import cache
    return cache.main(args.rest)


def cmd_train(args):
    """参数原样透传给 `radev.train.loop`（旋钮真值在 loop 的 parser）。"""
    from .train import loop
    return loop.main(args.rest)


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m radev', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    rc = sub.add_parser(
        'radar-cache', help='RADAR 冻结编码器逐 study 特征缓存',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='action（透传给 radev.train.cache）：\n'
                    '  plan   切分（CPU，幂等）\n'
                    '  run    跑第 --shard 片（--device cpu 可 CPU 冒烟）\n'
                    '全部旋钮见 `python -m radev.train.cache --help`。\n'
                    '真跑走 build_features.sh（GO=1 门控 + GPU 预检 + 盘上对账）。')
    rc.add_argument('rest', nargs=argparse.REMAINDER,
                    help='原样透传给 radev.train.cache 的参数（如 plan --ngpu 8）')
    rc.set_defaults(fn=cmd_radar_cache)

    tr = sub.add_parser(
        'train', help='projector + 小 LLM 训练 / 评估 / 地板',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='action（透传给 radev.train.loop）：\n'
                    '  train   bf16 + 梯度累积 + 检查点续跑（--rules 必给）\n'
                    '  eval    载 run dir 的 ckpt_last.pt，score_exact() 报数 + 地板对比\n'
                    '  floors  只算 majority / radar-cosine 两级地板（零训练）\n'
                    '全部旋钮见 `python -m radev.train.loop train --help`。')
    tr.add_argument('rest', nargs=argparse.REMAINDER,
                    help='原样透传给 radev.train.loop 的参数（如 train --rules a,b --tiny）')
    tr.set_defaults(fn=cmd_train)

    args = p.parse_args(argv)
    return args.fn(args)
