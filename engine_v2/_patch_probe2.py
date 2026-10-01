import io
p = r"F:\born-wired-cortex\engine_v2\tools\probe_retina_display.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = """    rows = np.asarray(eyes._rows, dtype=float)
    columns = np.asarray(eyes._columns, dtype=float)
    drawn_h, drawn_w = int(round(rows[-1])), int(round(columns[-1]))
    pickle_rows = np.clip(np.searchsorted(rows, np.arange(drawn_h), side="right") - 1,
                          0, eyes.height - 1)
    pickle_columns = np.clip(np.searchsorted(columns, np.arange(drawn_w), side="right") - 1,
                             0, eyes.width - 1)
    return np.asarray(sheet)[np.ix_(pickle_rows, pickle_columns)]"""
new = """    low_rows, high_rows = eyes._rows
    low_columns, high_columns = eyes._columns
    drawn_h, drawn_w = int(high_rows[-1]), int(high_columns[-1])
    which_rows = np.clip(np.searchsorted(low_rows, np.arange(drawn_h), side="right") - 1,
                         0, eyes.height - 1)
    which_columns = np.clip(np.searchsorted(low_columns, np.arange(drawn_w), side="right") - 1,
                            0, eyes.width - 1)
    return np.asarray(sheet)[np.ix_(which_rows, which_columns)]"""
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('print("格子边界（列，前 12 个）：", np.round(np.asarray(eyes._columns)[:12], 2).tolist())',
              'print("格子左边界（列，前 12 个）：", np.asarray(eyes._columns[0])[:12].tolist())')
s = s.replace('print("格子边界（列，后 6 个）：", np.round(np.asarray(eyes._columns)[-6:], 2).tolist())',
              'print("格子宽度（列，两端和中间）：", np.asarray(eyes._columns[1])[:4].tolist(), np.asarray(eyes._columns[1])[22:26].tolist(), np.asarray(eyes._columns[1])[-4:].tolist())')
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("改好了")