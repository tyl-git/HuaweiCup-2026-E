"""Native Qt desktop viewer for the existing Question 1 alignment results."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QFont, QKeySequence, QPalette, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDialog,
    QFrame, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSlider, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from src.gallery_model import GalleryStore, active_word, frame_index, valid_interval
from src.qt_widgets import VideoCanvas, TracePlot, VectorMap, WordStrip

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "progress-showcase/2026-09-23_visual-gallery-data_v1.0.json"

STYLE = """
QWidget { font-family:'Microsoft YaHei UI','Microsoft YaHei'; font-size:12px; color:#dce6f2; }
QMainWindow, QDialog { background:#0c121b; }
QFrame#card { background:#151e2a; border:1px solid #283547; border-radius:12px; }
QFrame#modality { background:#101822; border:1px solid #283547; border-radius:8px; }
QLabel { background:transparent; border:0; }
QLabel#title { font-size:22px; font-weight:700; color:#f3f6fc; }
QLabel#section { font-size:15px; font-weight:700; color:#eef4ff; }
QLabel#current { font-size:27px; font-weight:700; color:#75b6ff; }
QLabel#muted { color:#9aabc0; font-size:11px; }
QLabel#summary { color:#a9c4e7; background:#172639; padding:6px 12px; border-radius:7px; }
QLabel#notice { color:#deb873; }
QLabel#error { color:#ffb1a8; background:#38222a; padding:6px; border-radius:5px; }
QPushButton { background:#202e40; border:1px solid #344860; border-radius:6px; padding:6px 10px; }
QPushButton:hover { background:#2c4159; border-color:#6faee8; }
QPushButton:pressed { background:#304f75; }
QPushButton#primary { background:#326da8; border-color:#518ccc; color:white; }
QPushButton:disabled { color:#657488; border-color:#2a3545; background:#192330; }
QComboBox { background:#1b2a3b; border:1px solid #344860; border-radius:6px; padding:5px 9px; }
QComboBox QAbstractItemView { background:#1a2738; selection-background-color:#31557c; }
QSlider::groove:horizontal { height:5px; background:#2b3d53; border-radius:2px; }
QSlider::sub-page:horizontal { background:#6eafe7; }
QSlider::handle:horizontal { width:12px; margin:-4px 0; border-radius:6px; background:#a8d2ff; }
QSplitter::handle { background:#0c121b; }
QSplitter::handle:hover { background:#385577; }
QListWidget { background:#101822; border:0; border-radius:5px; outline:0; }
QListWidget::item { border:1px solid #294a46; border-radius:5px; padding:3px; }
QListWidget::item:selected { background:#28476a; border:2px solid #efc16c; color:#ffffff; }
QListWidget::item:hover { border-color:#83b7e3; }
QScrollBar:vertical { background:#142131; width:9px; margin:0; }
QScrollBar::handle:vertical { background:#435973; border-radius:4px; min-height:20px; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QTableWidget,QTextEdit { background:#121d2a; alternate-background-color:#192636; border:1px solid #30445c; }
QHeaderView::section { background:#243750; color:#dfebfa; border:0; padding:6px; }
QTabBar::tab { background:#1f3045; padding:8px 15px; }
QTabBar::tab:selected { background:#365777; }
QToolTip { background:#24354a; color:#f1f5fa; border:1px solid #547394; }
"""


def label(text="", name="", wrap=False):
    item = QLabel(text)
    item.setObjectName(name)
    item.setTextFormat(Qt.TextFormat.PlainText)
    item.setWordWrap(wrap)
    item.setMinimumWidth(0)
    return item


def card(name="card", margin=12):
    box = QFrame()
    box.setObjectName(name)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(6)
    return box, layout


def button(text, callback, primary=False):
    btn = QPushButton(text)
    if primary:
        btn.setObjectName("primary")
    btn.clicked.connect(callback)
    return btn


class MainWindow(QMainWindow):
    def __init__(self, store=None):
        super().__init__()
        self.store = store or GalleryStore(DEFAULT_DATA)
        self.sample = None
        self.features = None
        self.active_index = -2
        self.media_error = ""
        self.dialogs = []
        self.seeking = False
        self.was_playing = False
        self.setWindowTitle("华为杯 E 题 · 三模态词级对齐 · Qt 成果展示")
        self.setMinimumSize(1000, 640)
        self.resize(1340, 840)
        self.setStyleSheet(STYLE)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.6)
        self.player.setAudioOutput(self.audio_output)
        self._build_ui()
        self.player.setVideoSink(self.canvas.sink)
        self.player.positionChanged.connect(self.sync_position)
        self.player.playbackStateChanged.connect(self.play_state)
        self.player.errorOccurred.connect(self.play_error)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.mediaStatusChanged.connect(self.media_status)
        self.canvas.sink.videoFrameChanged.connect(self.frame_arrived)
        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(33)
        self.sync_timer.timeout.connect(lambda: self.sync_position(self.player.position()))
        self.sync_timer.start()
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_play)
        QShortcut(QKeySequence("Left"), self, activated=lambda: self.step_frame(-1))
        QShortcut(QKeySequence("Right"), self, activated=lambda: self.step_frame(1))
        self.refresh_picker()
        pilot = next((s["sample_id"] for s in self.store.samples if s["transcript_status"] == "confirmed_by_listening"), self.store.samples[0]["sample_id"])
        self.select_id(pilot)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 12, 16, 10)
        outer.setSpacing(8)
        head = QHBoxLayout()
        heading = QVBoxLayout()
        heading.setSpacing(0)
        heading.addWidget(label("三模态 · 同一时刻，同一个词", "title"))
        heading.addWidget(label("华为杯 E 题 / 问题一  ·  特征提取与时序对齐", "muted"))
        head.addLayout(heading, 1)
        s = self.store.summary
        stats = label(f"{s['sample_count']}/100 片段对齐   |   {s['source_words']:,} 原始词   |   {s['all_modalities_valid_words']:,} 三模态有效", "summary")
        head.addWidget(stats)
        head.addWidget(button("方法与质量说明", self.show_methods))
        outer.addLayout(head)
        toolbar = QHBoxLayout()
        toolbar.addWidget(label("片段"))
        self.picker = QComboBox()
        self.picker.setMinimumWidth(260)
        self.picker.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.picker.currentIndexChanged.connect(self.picker_changed)
        toolbar.addWidget(self.picker, 1)
        self.filter_box = QComboBox()
        for name in ("全部 100 个", "需要复核", "无有效视觉帧"):
            self.filter_box.addItem(name)
        self.filter_box.currentIndexChanged.connect(self.refresh_picker)
        toolbar.addWidget(self.filter_box)
        toolbar.addWidget(button("上一个", lambda: self.neighbor(-1)))
        toolbar.addWidget(button("下一个", lambda: self.neighbor(1)))
        toolbar.addWidget(button("65 词长片段", lambda: self.select_id("sample_0019")))
        toolbar.addWidget(button("视觉缺失示例", lambda: self.select_id("sample_0010")))
        outer.addLayout(toolbar)
        self.sample_note = label("", "notice")
        self.sample_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        outer.addWidget(self.sample_note)
        self.main_split = QSplitter(Qt.Orientation.Horizontal)
        self.main_split.setChildrenCollapsible(False)
        self.main_split.setHandleWidth(8)
        outer.addWidget(self.main_split, 1)
        self.left_split = QSplitter(Qt.Orientation.Vertical)
        self.left_split.setChildrenCollapsible(False)
        self.left_split.setHandleWidth(8)
        self.main_split.addWidget(self.left_split)
        self.video_card, vl = card()
        self.video_card.setMinimumWidth(420)
        video_title = QHBoxLayout()
        video_title.addWidget(label("01  原始视频", "section"), 1)
        self.eyes = QCheckBox("眼部点位")
        self.eyes.setChecked(True)
        video_title.addWidget(self.eyes)
        vl.addLayout(video_title)
        self.canvas = VideoCanvas()
        self.eyes.toggled.connect(self.canvas.set_landmarks)
        vl.addWidget(self.canvas, 1)
        seek_row = QHBoxLayout()
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.sliderPressed.connect(self.seek_begin)
        self.seek.sliderReleased.connect(self.seek_end)
        self.seek.sliderMoved.connect(self.seek_preview)
        self.seek.valueChanged.connect(self.seek_changed)
        seek_row.addWidget(self.seek, 1)
        self.clock = label("0.00 / 0.00 s", "muted")
        self.clock.setMinimumWidth(98)
        seek_row.addWidget(self.clock)
        vl.addLayout(seek_row)
        controls = QHBoxLayout()
        self.play_button = button("播放", self.toggle_play, True)
        controls.addWidget(self.play_button)
        controls.addWidget(button("◀ 帧", lambda: self.step_frame(-1)))
        controls.addWidget(button("帧 ▶", lambda: self.step_frame(1)))
        self.speed = QComboBox()
        for value in (0.5, 0.75, 1.0, 1.25):
            self.speed.addItem(f"{value:g} 倍速", value)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(lambda: self.player.setPlaybackRate(self.speed.currentData()))
        controls.addWidget(self.speed)
        controls.addStretch(1)
        self.mute = QCheckBox("静音")
        self.mute.toggled.connect(self.audio_output.setMuted)
        controls.addWidget(self.mute)
        vl.addLayout(controls)
        self.frame_label = label("等待视频载入", "muted")
        vl.addWidget(self.frame_label)
        self.error_label = label("", "error", True)
        self.error_label.hide()
        vl.addWidget(self.error_label)
        self.left_split.addWidget(self.video_card)
        self.timeline_card, tl = card()
        # Keep the timeline compact so the video and all three modality panels
        # remain visible together on a laptop-sized window.  The splitter is
        # still user-adjustable when a longer transcript needs more room.
        self.timeline_card.setMinimumHeight(156)
        self.timeline_card.setMaximumHeight(250)
        self.timeline_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        th = QHBoxLayout()
        th.addWidget(label("02  词语时间轴", "section"), 1)
        self.word_count = label("", "muted")
        th.addWidget(self.word_count)
        tl.addLayout(th)
        self.strip = WordStrip()
        self.strip.setFixedHeight(58)
        self.strip.wordClicked.connect(self.jump_word)
        tl.addWidget(self.strip)
        tl.addWidget(label("轨道：文字 / 声音 / 画面 · 点击词跳转 · 绿色齐全 / 黄色缺失 / 灰色无时间", "muted"))
        self.word_list = QListWidget()
        self.word_list.setViewMode(QListView.ViewMode.IconMode)
        self.word_list.setFlow(QListView.Flow.LeftToRight)
        self.word_list.setWrapping(True)
        self.word_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.word_list.setMovement(QListView.Movement.Static)
        self.word_list.setGridSize(QSize(109, 44))
        self.word_list.setSpacing(3)
        self.word_list.setMinimumHeight(46)
        self.word_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.word_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.word_list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.word_list.itemClicked.connect(lambda item: self.jump_word(item.data(Qt.ItemDataRole.UserRole)))
        self.word_list.itemActivated.connect(lambda item: self.jump_word(item.data(Qt.ItemDataRole.UserRole)))
        tl.addWidget(self.word_list, 1)
        self.left_split.addWidget(self.timeline_card)
        self.sync_card, sl = card()
        self.sync_card.setMinimumWidth(380)
        sync_title = QHBoxLayout()
        sync_title.addWidget(label("03  同步查看三模态", "section"), 1)
        self.details_button = button("当前词全部数值", self.show_features)
        sync_title.addWidget(self.details_button)
        sl.addLayout(sync_title)
        active_row = QHBoxLayout()
        self.current_word = label("当前无词", "current")
        self.current_word.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        active_row.addWidget(self.current_word, 1)
        self.current_time = label("—", "muted")
        active_row.addWidget(self.current_time)
        sl.addLayout(active_row)
        self.current_note = label("", "muted")
        self.current_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        sl.addWidget(self.current_note)
        text_box, tex = card("modality", 9)
        th = QHBoxLayout()
        th.addWidget(label("文字  /  BERT · 768 维", "section"), 1)
        self.text_state = label("—")
        th.addWidget(self.text_state)
        tex.addLayout(th)
        self.vector = VectorMap()
        self.vector.setMinimumHeight(48)
        self.vector.setMaximumHeight(110)
        tex.addWidget(self.vector, 1)
        self.vector_legend = label("完整词向量 · 蓝负 / 橙正 · 不是情绪强度", "muted")
        tex.addWidget(self.vector_legend)
        sl.addWidget(text_box, 1)
        self.audio_card, au = card("modality", 9)
        ah = QHBoxLayout()
        ah.addWidget(label("声音  /  eGeMAPS · 25 维", "section"), 1)
        self.audio_state = label("—")
        ah.addWidget(self.audio_state)
        au.addLayout(ah)
        self.audio_values = label("词内响度 — · 音高 —", "muted")
        au.addWidget(self.audio_values)
        self.audio_plot = TracePlot()
        au.addWidget(self.audio_plot, 1)
        au.addWidget(label("曲线：逐帧响度（sone） · 色带：当前词区间", "muted"))
        sl.addWidget(self.audio_card, 1)
        self.vision_card, vi = card("modality", 9)
        vh = QHBoxLayout()
        vh.addWidget(label("画面  /  OpenFace · 49 维", "section"), 1)
        self.vision_state = label("—")
        vh.addWidget(self.vision_state)
        vi.addLayout(vh)
        self.vision_values = label("词内嘴角 — · 双唇 — · 眨眼 —", "muted")
        vi.addWidget(self.vision_values)
        self.vision_plot = TracePlot()
        vi.addWidget(self.vision_plot, 1)
        vi.addWidget(label("曲线：逐帧双唇分离 AU25（0–5） · 无效帧留空", "muted"))
        sl.addWidget(self.vision_card, 1)
        sl.addWidget(label("数值是当前词区间的汇聚特征；动作强度不能直接当作情绪结论。", "muted", True))
        self.main_split.addWidget(self.sync_card)
        # The left column is intentionally close to square: a 16:9 video sits
        # above a compact word timeline whose width exactly matches the video.
        # Give the video the larger share by default while retaining both
        # splitters for manual resizing.
        self.main_split.setSizes([800, 490])
        self.left_split.setSizes([470, 210])
        foot = label(f"{s['needs_review_samples']} 个片段带复核标记 · {s['no_valid_visual_samples']} 个无有效视觉帧仍保留 · Space 播放/暂停 · ←/→ 逐帧 · 拖动中间分隔线调整区域", "muted")
        outer.addWidget(foot)

    def refresh_picker(self, *_):
        old = self.sample["sample_id"] if self.sample else None
        mode = self.filter_box.currentIndex()
        samples = [s for s in self.store.samples if mode == 0 or (mode == 1 and s['multimodal']['needs_review']) or (mode == 2 and s['valid_frames'] == 0)]
        with QSignalBlocker(self.picker):
            self.picker.clear()
            for sample in samples:
                name = f"{sample['sample_id']}  ·  {sample['duration_seconds']:.1f}s  ·  {len(sample['multimodal']['words'])} 词"
                self.picker.addItem(name, sample['sample_id'])
            idx = self.picker.findData(old)
            self.picker.setCurrentIndex(max(0, idx))
        if self.picker.count():
            self.picker_changed()

    def picker_changed(self, *_):
        sid = self.picker.currentData()
        if sid and (self.sample is None or sid != self.sample['sample_id']):
            self.load_sample(sid)

    def select_id(self, sid):
        if sid not in self.store.by_id:
            return
        if self.picker.findData(sid) < 0:
            self.filter_box.setCurrentIndex(0)
        self.picker.setCurrentIndex(self.picker.findData(sid))
        if self.sample is None or self.sample['sample_id'] != sid:
            self.load_sample(sid)

    def neighbor(self, delta):
        self.picker.setCurrentIndex((self.picker.currentIndex() + delta) % self.picker.count())

    def load_sample(self, sid):
        self.player.stop()
        self.player.setSource(QUrl())
        self.canvas.clear()
        self.media_error = ""
        self.error_label.hide()
        self.sample = self.store.by_id[sid]
        self.features = None
        self.active_index = -2
        sample, mm = self.sample, self.store.by_id[sid]['multimodal']
        self.canvas.set_sample(sample)
        self.strip.set_sample(sample)
        n, complete = len(mm['words']), sum(mm['all_modalities_mask'])
        note = f"{sample['source_video_id']} / {sample['clip_id']}  ·  三模态有效 {complete}/{n} 词  ·  "
        note += "带复核标记（不等于失败）" if mm['needs_review'] else "当前无质量复核标记"
        if sample['transcript_note']:
            note += " · 字幕存在已知疑点"
        self.sample_note.setText(note)
        self.sample_note.setToolTip(note + "\n" + sample['transcript_note'] + "\n" + sample['transcript'])
        self.word_count.setText(f"{n} 词 / {complete} 三模态有效")
        self.populate_words()
        self.seek.setRange(0, round(sample['duration_seconds'] * 1000))
        trace = sample['audio_trace']
        offset = sample.get('word_video_offset_seconds', 0)
        self.audio_plot.set_series([t + offset for t in trace['times']], trace['loudness'], sample['duration_seconds'], '#58b895')
        self.vision_plot.set_series([t + sample['time_offset_seconds'] for t in sample['timestamps']], sample['au25'], sample['duration_seconds'], '#dc9d4b', 5)
        try:
            self.features = self.store.features(sample)
        except (OSError, ValueError) as exc:
            self.error_label.setText("合并特征读取失败：" + str(exc))
            self.error_label.show()
        self.sync_position(0)
        try:
            self.player.setSource(QUrl.fromLocalFile(str(self.store.video_path(sample))))
            self.play_button.setEnabled(True)
        except (OSError, ValueError) as exc:
            self.play_error(None, str(exc))

    def populate_words(self):
        self.word_list.clear()
        mm = self.sample['multimodal']
        for i, word in enumerate(mm['words']):
            pair = valid_interval(self.sample, i)
            time_text = f"{pair[0]:.2f}–{pair[1]:.2f}s" if pair else "无有效时间"
            item = QListWidgetItem(f"{word}\n{time_text}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setSizeHint(QSize(103, 40))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            valid = mm['all_modalities_mask'][i]
            item.setBackground(QColor('#173a34' if valid else '#3c3022' if pair else '#26303b'))
            item.setForeground(QColor('#bee2d3' if valid else '#e5c68f' if pair else '#929dab'))
            flags = mm['word_review_flags'][i]
            item.setToolTip(f"{word} · {time_text}\n文字 {'有效' if mm['text_mask'][i] else '缺失'} / 声音 {'有效' if mm['audio_mask'][i] else '缺失'} / 画面 {'有效' if mm['vision_mask'][i] else '缺失'}\n{flags}")
            if pair is None:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.word_list.addItem(item)

    def duration_changed(self, duration):
        if self.sample:
            self.seek.setMaximum(max(duration, round(self.sample['duration_seconds'] * 1000)))

    def media_status(self, status):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self.player.setPlaybackRate(self.speed.currentData())
            self.sync_position(self.player.position())

    def play_error(self, error, message):
        self.media_error = message
        self.error_label.setText("视频播放失败：" + message)
        self.error_label.show()
        self.play_button.setEnabled(False)

    def play_state(self, state):
        self.play_button.setText("暂停" if state == QMediaPlayer.PlaybackState.PlayingState else "播放")

    def toggle_play(self):
        if self.media_error:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            if self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
                self.player.setPosition(0)
            self.player.play()

    def seek_begin(self):
        self.seeking = True
        self.was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self.player.pause()

    def seek_preview(self, value):
        self.player.setPosition(value)
        self.sync_position(value)

    def seek_end(self):
        self.player.setPosition(self.seek.value())
        self.seeking = False
        if self.was_playing:
            self.player.play()

    def seek_changed(self, value):
        if not self.seeking:
            self.player.setPosition(value)
            self.sync_position(value)

    def jump_word(self, index):
        pair = valid_interval(self.sample, index)
        if not pair:
            return
        # Ceil prevents integer-millisecond rounding from landing before the word.
        position = math.ceil(pair[0] * 1000)
        self.player.setPosition(position)
        self.sync_position(position)
        self.player.play()

    def step_frame(self, delta):
        if not self.sample:
            return
        self.player.pause()
        idx = frame_index(self.sample, self.player.position() / 1000)
        if idx < 0:
            idx = 0 if self.player.position() == 0 else self.sample['frame_count'] - 1
        idx = max(0, min(self.sample['frame_count'] - 1, idx + delta))
        t = self.sample['timestamps'][idx] + self.sample['time_offset_seconds']
        position = math.ceil(t * 1000)
        self.player.setPosition(position)
        self.sync_position(position)

    def frame_arrived(self, frame):
        if frame.isValid() and self.sample and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState and frame.startTime() >= 0:
            self.sync_position(frame.startTime() // 1000)

    def sync_position(self, milliseconds):
        if not self.sample:
            return
        seconds = max(0, milliseconds / 1000)
        with QSignalBlocker(self.seek):
            if not self.seeking:
                self.seek.setValue(milliseconds)
        self.clock.setText(f"{seconds:.2f} / {self.sample['duration_seconds']:.2f} s")
        self.canvas.set_time(seconds)
        self.strip.set_time(seconds)
        index = active_word(self.sample, seconds)
        interval = valid_interval(self.sample, index) if index >= 0 else None
        self.audio_plot.set_cursor(seconds, interval)
        self.vision_plot.set_cursor(seconds, interval)
        idx = frame_index(self.sample, seconds)
        if idx >= 0:
            quality = '有效' if self.sample['valid'][idx] else '无有效视觉'
            self.frame_label.setText(f"第 {idx + 1}/{self.sample['frame_count']} 帧  ·  {quality}  ·  置信度 {self.sample['confidence'][idx]:.2f}（阈值 0.80）")
        else:
            self.frame_label.setText("片段结束" if seconds >= self.sample['duration_seconds'] else "首帧尚未开始")
        if index == self.active_index:
            return
        self.active_index = index
        self.update_word(index)

    def update_word(self, index):
        mm = self.sample['multimodal']
        self.word_list.clearSelection()
        self.details_button.setEnabled(index >= 0 and self.features is not None)
        if index < 0:
            self.current_word.setText("当前无词 · 停顿")
            self.current_time.setText("—")
            self.current_note.setText("播放或点击词语；有发音时间时同步显示三路特征。")
            self.vector.set_vector(None)
            self.vector_legend.setText("完整词向量 · 蓝负 / 橙正 · 不是情绪强度")
            for state in (self.text_state, self.audio_state, self.vision_state):
                state.setText("当前无词")
                state.setStyleSheet("color:#98a5b6")
            self.audio_values.setText("词内响度 — · 音高 —")
            self.vision_values.setText("词内嘴角 — · 双唇 — · 眨眼 —")
            return
        item = self.word_list.item(index)
        self.word_list.setCurrentItem(item)
        item.setSelected(True)
        self.word_list.scrollToItem(item, QAbstractItemView.ScrollHint.EnsureVisible)
        self.current_word.setText(mm['words'][index])
        self.current_word.setToolTip(mm['words'][index])
        a, b = valid_interval(self.sample, index)
        self.current_time.setText(f"{a:.2f}–{b:.2f} 秒")
        note = f"第 {index + 1}/{len(mm['words'])} 词 · " + ("三模态同时有效" if mm['all_modalities_mask'][index] else "存在缺失模态，已保留掩码")
        if mm['word_review_flags'][index]:
            note += " · 此词带复核标记"
        self.current_note.setText(note)
        self.current_note.setToolTip(note + '\n' + mm['word_review_flags'][index])
        for key, state in [('text', self.text_state), ('audio', self.audio_state), ('vision', self.vision_state)]:
            ok = mm[key + '_mask'][index]
            state.setText("有效" if ok else "缺失")
            state.setStyleSheet("color:#74d1ae" if ok else "color:#e7b367")
        self.vector.set_vector(self.features['text'][index] if self.features is not None and mm['text_mask'][index] else None)
        if self.features is None:
            self.text_state.setText("向量文件不可用")
        self.vector_legend.setText(f"蓝负 / 橙正 · 色阶 ±{self.vector.max_abs:.2f}（当前词） · 非情绪强度")
        if mm['audio_mask'][index]:
            loud = mm['audio_loudness'][index]
            pitch = mm['audio_pitch_semitone'][index]
            self.audio_values.setText(f"词内响度 {loud:.3f} sone · 音高 {pitch:.2f} st · 覆盖 {mm['audio_window_coverage'][index]:.0%}")
            self.audio_values.setToolTip("音高单位 st：相对 27.5 Hz 的半音数；0 可能是无声或未检测到音高。词内值为时间重叠加权汇聚。")
        else:
            self.audio_values.setText("该词无有效音频特征 · 占位零不作为测量值")
        if mm['vision_mask'][index]:
            self.vision_values.setText(f"嘴角 {mm['vision_au12'][index]:.2f} · 双唇 {mm['vision_au25'][index]:.2f} · 眨眼 {mm['vision_au45'][index]:.2f} · 覆盖 {mm['vision_coverage'][index]:.0%}")
        else:
            self.vision_values.setText("该词无有效视觉特征 · 占位零不作为测量值")

    def show_features(self):
        if self.active_index < 0 or self.features is None:
            return
        self.player.pause()
        idx = self.active_index
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle(f"{self.sample['sample_id']} / {self.sample['multimodal']['words'][idx]} · 合并特征原值")
        dialog.resize(680, 570)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("来自最终合并 NPZ；缺失模态显示“缺失”，不将占位零显示为有效测量。", "muted", True))
        tabs = QTabWidget()
        for key, title in [('text', '文字 · 768'), ('audio', '声音 · 25'), ('vision', '画面 · 49')]:
            names = self.features['feature_names'][key]
            table = QTableWidget(len(names), 2)
            table.setHorizontalHeaderLabels(['特征', '当前词数值'])
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setAlternatingRowColors(True)
            ok = self.sample['multimodal'][key + '_mask'][idx]
            for row, (name, value) in enumerate(zip(names, self.features[key][idx])):
                table.setItem(row, 0, QTableWidgetItem(str(name)))
                table.setItem(row, 1, QTableWidgetItem(f"{value:.7g}" if ok else '缺失（mask=false）'))
            table.setColumnWidth(0, 375)
            table.horizontalHeader().setStretchLastSection(True)
            tabs.addTab(table, title)
        layout.addWidget(tabs)
        self.dialogs.append(dialog)
        dialog.finished.connect(lambda: self.dialogs.remove(dialog))
        dialog.show()

    def show_methods(self):
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("特征提取与时序对齐 · 方法与质量")
        dialog.resize(700, 540)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(
            "这是问题一的特征提取与时序对齐展示。\n\n"
            "文字：BERT 提取上下文 token 向量，同一原始词的子词向量汇聚为 768 维。热图显示完整向量，蓝负橙正，色阶按当前词绝对最大值；没有把向量维度解释为情绪。\n\n"
            "声音：openSMILE eGeMAPSv02 低层描述符，25 维。界面预览响度和音高；响度为 sone，音高 st 为相对 27.5 Hz 的半音数，0 也可能是无声或未检测出音高。\n\n"
            "画面：OpenFace，MTCNN 检测、CLNF 跟踪；49 维包含视线8维、头部姿态6维、AU强度17维和AU出现18维。界面曲线为AU25双唇分离，眼部点位为原始逐帧24点（不属于全部49维）。当前流程跟踪一个检测人脸，没有验证其就是发声者。\n\n"
            "时间：Wav2Vec2 CTC 强制对齐给出原始词区间；音频窗口和视觉帧按与词区间的重叠时长加权汇聚。播放时依据同一个视频时间更新词、轨道和曲线。停顿不延续上一词特征。\n\n"
            "质量：有效视觉帧要求success=1、confidence≥0.8。9个无定时词保留但不能跳转；4个无有效视觉帧样本保留；缺失模态用mask标记。65个样本带复核标记，并不表示对齐失败。仅典型样本人工确认了字幕及整体停顿/语序/语速，精确词边界未全部人工核验；已知字幕疑点继续保留。\n\n"
            "操作：空格播放/暂停；左右方向键逐帧；0.5/0.75倍速听辨；点击词跳转播放。可拖动区域分隔线。长句只在词语列表内滚动。全量数值从最终合并NPZ读取。\n\n"
            f"数据：{self.store.data_path}"
        )
        layout.addWidget(text)
        self.dialogs.append(dialog)
        dialog.finished.connect(lambda: self.dialogs.remove(dialog))
        dialog.show()

    def closeEvent(self, event):
        self.sync_timer.stop()
        self.player.stop()
        self.player.setSource(QUrl())
        super().closeEvent(event)


def create_application():
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    app.setFont(QFont('Microsoft YaHei UI', 9))
    palette = QPalette()
    for role, color in [(QPalette.ColorRole.Window, '#0c121b'), (QPalette.ColorRole.WindowText, '#dce6f2'), (QPalette.ColorRole.Base, '#141f2e'), (QPalette.ColorRole.Text, '#dce6f2'), (QPalette.ColorRole.Button, '#243750'), (QPalette.ColorRole.ButtonText, '#dce6f2'), (QPalette.ColorRole.Highlight, '#345e89')]:
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--audit', action='store_true', help='Validate all source features; do not open a window')
    args = parser.parse_args()
    store = GalleryStore(args.data)
    if args.audit:
        result = store.audit()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(bool(result['errors']))
    app = create_application()
    try:
        window = MainWindow(store)
    except Exception as exc:
        QMessageBox.critical(None, '启动失败', str(exc))
        return 1
    area = app.primaryScreen().availableGeometry()
    window.resize(min(1340, area.width() - 30), min(840, area.height() - 45))
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
