"""Frozen P70 event policy from the original study protocol."""
import numpy as np

from .settings import TASK, THRESHOLD

QUERY, TARGET = TASK.anchors, TASK.targets
ANCHOR_INDICES = TASK.anchor_offsets
NONANCHORS = TARGET[~np.isin(TARGET, QUERY)]


def policy(anchors):
    """Use predicted anchor currents only. Return both selections with identical per-record budgets."""
    assert anchors.ndim==2 and anchors.shape[1]==len(QUERY) and np.isfinite(anchors).all()
    inside_window = np.abs(anchors[:,1:]) < THRESHOLD
    initial = anchors[:,0] <= -THRESHOLD
    trigger = initial & inside_window.any(axis=1)
    first = inside_window.argmax(axis=1)+1
    left = np.where(trigger,QUERY[first-1],-1)
    right = np.where(trigger,QUERY[first],-1)
    selected_exit = (TARGET[None,:]>left[:,None]) & (TARGET[None,:]<right[:,None]) & trigger[:,None]
    selected_start = (np.cumsum(np.isin(TARGET,NONANCHORS))[None,:] <= selected_exit.sum(axis=1)[:,None]) & np.isin(TARGET,NONANCHORS)
    assert np.array_equal(selected_exit.sum(axis=1),selected_start.sum(axis=1))
    assert not (selected_exit[:,ANCHOR_INDICES].any() or selected_start[:,ANCHOR_INDICES].any())
    reason = np.select([trigger,initial],['exit_bracket','no_anchor_exit'],default='no_initial_negative_plateau')
    return dict(trigger=trigger,reason=reason,left=left,right=right,Exit=selected_exit,Start=selected_start)


def exit_index(curves):
    """Original plateau exit definition. -1 indicates no in-window negative-platform exit."""
    assert curves.ndim==2 and curves.shape[1]==len(TARGET)
    outside = np.abs(curves) < THRESHOLD
    return np.where((curves[:,0]<=-THRESHOLD)&outside.any(1),TARGET[outside.argmax(1)],-1)
