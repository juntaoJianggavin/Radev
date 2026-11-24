"""唯一配置点：数据源路径、版本常量、stage 0 的期望口径、阈值。

所有 EXPECT_* 常量都是实测值（见 Jev/docs/数据处理计划.md §14 的复现命令）。
stage 0 加载后逐一断言，不符即中止 —— 上游文件被换过而我们不知道，是最危险的静默错误。
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('RADEV_IMAGE_ROOT') or ROOT
REPORTS = f'{DATA}/reports_final.xlsx'
METADATA = f'{DATA}/metadata.csv'
ZERO_SHOT = f'{DATA}/zero_shot_findings_disease_cls.csv'
FIVE_YEARS = f'{DATA}/five_years_disease_task.csv'
IMAGE_DIR = 'iid'


def image_path(sid, vol=None):
    if vol:
        return vol if os.path.isabs(vol) else os.path.join(DATA, vol)
    sub = 'ood' if sid.startswith('ood_') else IMAGE_DIR
    return os.path.join(DATA, sub, f'{sid}.nii.gz')
OUT = os.environ.get('JEV_OUT') or os.path.join(ROOT, 'out_v3')
if not os.path.isabs(OUT):
    OUT = os.path.join(ROOT, OUT)
AUDIT = os.path.join(OUT, 'audit')
STAGING = os.path.join(OUT, 'staging')

BASELINE_V2 = os.path.join(ROOT, 'out', 'baseline_v2.jsonl')

NEWQ = os.path.join(ROOT, 'out_newq')

REVIEW = os.path.join(ROOT, 'out_review')
REVIEW_DEFAULT = REVIEW

LABELER = 'rule_v4'
SCHEMA_VERSION = '2.0'
ROUTE = 't2'

EXPECT_REPORT_ROWS = 25494
EXPECT_REPORT_UNIQUE = 25489
EXPECT_METADATA_ROWS = 25414
EXPECT_METADATA_UNIQUE = 25412
EXPECT_FIVE_YEARS_ROWS = 12416
EXPECT_FIVE_YEARS_UNIQUE = 12353
EXPECT_ZERO_SHOT_ROWS = 25275
EXPECT_IMAGE_FILES = 25489

EXPECT_NO_METADATA = 77

EXCLUDED_STUDIES = {
    'AC4242d57': 'two unrelated reports under one study id; single nii.gz makes '
                 'the image-report pairing unknowable, so every label is suspect',
}
EXPECT_AFTER_EXCLUDE = EXPECT_REPORT_UNIQUE - len(EXCLUDED_STUDIES)

DUPLICATE_KEEP_FIRST = ('AC423c4f0', 'AC4214d72')

SPLITS = ('train', 'val', 'test')
EXPECT_SPLIT = {'train': 15309, 'val': 5055, 'test': 5125}
EXPECT_SPLIT_AFTER_EXCLUDE = {'train': 15308, 'val': 5055, 'test': 5125}

EXPECT_HEADER_STUDIES = 22135
EXPECT_NO_HEADER_STUDIES = 3353

EXPECT_SUMMARY = 24267
EXPECT_LAST_SECTION_SUMMARY = 21210

EXPECT_SUMMARY_FLIPS = 8812
EXPECT_POS_NET = -150
EXPECT_POS_DELTA = {
    'fracture': -90, 'metastatic_disease': -25, 'ascites': -15, 'osteopenia': -13,
    'lymphadenopathy': -3, 'anasarca': -2, 'bowel_obstruction': -1, 'free_air': -1,
}

ADJ_WINDOW = 5
SEV_LEVELS = ('minimal', 'mild', 'moderate', 'severe', 'massive')
SIZE_TERMS = ('small', 'large', 'tiny')

SCORE_MIN_LEVELS = 2
SCORE_LEVEL_COUNT = len(SEV_LEVELS)

SCORE_POSITIVE_ONLY = True

PROGNOSIS_CLASSES = (0, 1)
EXPECT_PROG_TRAIN = 38412
EXPECT_PROG_EVAL = 9702

ACQ_EMIT_CONTRAST = False
EXPECT_ACQ_PHASE = 25405

EXPECT_TRIAGE = 5673
EXPECT_TRIAGE_ACUTE = 1915
EXPECT_TRIAGE_NOT_ACUTE = 3758
EXPECT_TRIAGE_BOTH = 150
