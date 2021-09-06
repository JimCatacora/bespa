import numpy as np
 
def get_dist_mr_to_stride(idx, stride, st_delta, stdir, maxidx):
  d = 0
  if stdir == 'stay':
    return d
  elif stdir == 'left':
    d += st_delta
    if stride > idx:
      acts = maxidx - stride
      d += 2 * acts
  else:
    acts = (stride if stride >= idx else maxidx) - idx
    d += 2 * acts
    d += st_delta
  return d
 
def get_dist_mr_at_edge(goal, start, rval, stdir, carry, maxclr):
  if stdir == 'stay':
    if rval == 0:
      g = goal
      d1 = g - start
      d2 = maxclr - g + start + 4
    else:
      g = goal
      d1 = abs(g - start) + 3
      d2 = maxclr - g + start + 1
  elif stdir == 'left':  
    if rval == 0:
      g = goal if not carry else (maxclr + 1 + goal)
      d1 = g - start
      d2 = abs(maxclr + 1 - g + start) + 3
    else:
      g = goal if not carry else (maxclr + 1 + goal)
      d1 = abs(g - start) + 3
      d2 = abs(maxclr - g + start + 1)
  else:
    if rval == 0:
      g = goal
      d1 = abs(g - start) + 2
      d2 = maxclr - g + start + 2
    else:
      g = goal if not carry else -1
      d1 = abs(g - start) + 3
      d2 = maxclr - g + start + 3
  if d1 <= d2:
    return d1
  else:
    return d2
 
def get_dist_mr_stride_dir(station, stride, st_delta, stdir, carry, maxidx, maxclr, done, best_d):
  doneidx, doneval = done
  tv = [0]
  ds = [0]
  for i, s in enumerate(station):
    n = len(tv)
    if n == 0:
      break
    for _ in range(n):
      v = tv.pop(0)
      d = ds.pop(0)
      #goup
      if v == 1:
        if i == doneidx and i < maxidx:
          egd = get_dist_mr_at_edge(s, v, doneval, stdir, carry, maxclr)
          std = get_dist_mr_to_stride(i + (1 if stdir == 'right' else 0), stride, st_delta, stdir, maxidx)
          total_d = d + egd + std
          if total_d < best_d:
            best_d = total_d
        elif s > 0:
          if i == maxidx:
            std = get_dist_mr_to_stride(i, stride, st_delta, stdir, maxidx)
            total_d = d + s - 1 + std
            if total_d < best_d:
              best_d = total_d
          else:
            new_d = d + s
            if new_d < best_d:
              tv.append(1)
              ds.append(new_d)
      elif n == 1 and (i < doneidx or doneidx == maxidx):
        if s > 0:
          if i == maxidx:
            std = get_dist_mr_to_stride(i, stride, st_delta, stdir, maxidx)
            total_d = d + s + std
            if total_d < best_d:
              best_d = total_d
          else:
            new_d = d + s + 1
            if new_d < best_d:
              tv.append(1)
              ds.append(new_d)
      #godown
      if v == 0:
        if i == doneidx and i < maxidx:
          egd = get_dist_mr_at_edge(s, v, doneval, stdir, carry, maxclr)
          std = get_dist_mr_to_stride(i + (1 if stdir == 'right' else 0), stride, st_delta, stdir, maxidx)
          total_d = d + egd + std
          if total_d < best_d:
            best_d = total_d
        elif s > 0:
          if i == maxidx:
            std = get_dist_mr_to_stride(i, stride, st_delta, stdir, maxidx)
            total_d = d + maxclr - s + 1 + std
            if total_d < best_d:
              best_d = total_d
          else:
            new_d = d + maxclr - s + 2
            if new_d < best_d:
              tv.append(0)
              ds.append(new_d)
        else:
          if i == maxidx:
            std = get_dist_mr_to_stride(i, stride, st_delta, stdir, maxidx)
            total_d = d + std
            if total_d < best_d:
              best_d = total_d
          else:
            new_d = d + 1
            if new_d < best_d:
              tv.append(1)
              ds.append(new_d)
      elif n == 1 and (i < doneidx or doneidx == maxidx):
        if s > 1:
          if i == maxidx:
            std = get_dist_mr_to_stride(i, stride, st_delta, stdir, maxidx)
            total_d = d + maxclr - s + 2 + std
            if total_d < best_d:
              best_d = total_d
          else:
            new_d = d + maxclr - s + 3
            if new_d < best_d:
              tv.append(0)
              ds.append(new_d)
        elif s == 0:
          if i == maxidx:
            std = get_dist_mr_to_stride(i, stride, st_delta, stdir, maxidx)
            total_d = d + 1 + std
            if total_d < best_d:
              best_d = total_d
          else:
            new_d = d + 2
            if new_d < best_d:
              tv.append(1)
              ds.append(new_d)
    if len(ds) > 1:
      if ds[0] != ds[1]:
        deli = ds.index(max(ds))
        del tv[deli]
        del ds[deli]
  return best_d
 
def get_distance_moving_right(station, stride, colors, best_d, stdir='both'):
  stnlen = len(station)
  maxidx = stnlen - 1
  maxclr = colors - 1
  if all([s == 0 for s in station]):
    d1 = stride
    d2 = stnlen - stride
    if d1 <= d2:
      return d1 * 2
    else:
      return d2 * 2
  elif all([s == maxclr for s in station]):
    d1 = stride
    d2 = maxidx - stride
    if d1 <= d2:
      return d1 * 2 + 1
    else:
      return d2 * 2 + 1

  doneval = station[-1]
  if doneval in [0, maxclr]:
    doneidx = 0
    for s in reversed(station):
      if s == doneval:
        doneidx += 1
      else:
        break
    doneidx = maxidx - doneidx
  else:
    doneidx = maxidx
  if stride == doneidx:
    best_d = get_dist_mr_stride_dir(station, stride, 0, 'stay', 0, maxidx, maxclr, (doneidx, doneval), best_d)
  else:
    #stride_right
    if stdir in ['both', 'right']:
      if stride < doneidx:
        st_delta = stride + 1
        adj_station = []
        c = 0
        carry = 0
        for i, s in enumerate(station):
          rep = i <= stride
          if not rep and c == 0:
            adj_station.extend(station[i:])
            break
          offset = 1 if rep else 0
          adj_s = s - offset - c
          if adj_s < 0:
            adj_s += colors
            c = 1
          else:
            c = 0
          if i == doneidx:
            carry = c
          adj_station.append(adj_s)
      elif stride > doneidx:
        st_delta = 0
        carry = 0
        adj_station = station[:]
      best_d = get_dist_mr_stride_dir(adj_station, stride, st_delta, 'right', carry, maxidx, maxclr, (doneidx, doneval), best_d)
    #stride_left
    if stdir in ['both', 'left']:
      steq = stride if stride < doneidx else -1
      st_delta = doneidx - steq
      adj_station = []
      c = 0
      carry = 0
      for i, s in enumerate(station):
        rep = i > steq and i <= doneidx
        offset = 1 if rep else 0
        adj_s = s + offset + c
        if adj_s > maxclr:
          adj_s -= colors
          c = 1
        else:
          c = 0
        if i == doneidx:
          carry = c
        adj_station.append(adj_s)
        if i >= doneidx and c == 0:
          adj_station.extend(station[i + 1:])
          break
      best_d = get_dist_mr_stride_dir(adj_station, stride, st_delta, 'left', carry, maxidx, maxclr, (doneidx, doneval), best_d)
  return best_d
  
def get_dist_ml_to_stride(idx, stride, st_delta, stdir, extra=0):
  d = 0
  if stdir == 'left':
    acts = idx - (stride + 1 if stride < idx else 1)
    d += 2 * acts
    d += st_delta
  else:
    d += st_delta
    acts = stride if stride > 0 and stride < idx else 0
    if extra > 0:
      d += extra + 2 * (acts - 1)
    else:
      d += 2 * acts
  return d
 
def get_dist_ml_stride_dir(station, stride, st_delta, stdir, initvdx, maxidx, maxclr, doneidx, best_d, extra=0):
  v0, d0 = initvdx
  done = maxidx - doneidx
  off = [0]
  ds = [d0]
  if v0 == 0:
    for i, s in enumerate(reversed(station[1:])):
      n = len(off)
      if n == 0:
        break
      o1 = o2 = off.pop(0)
      d1 = d2 = ds.pop(0)
      if n > 1:
        o2 = off.pop(0)
        d2 = ds.pop(0)
      if i == done:
        std = get_dist_ml_to_stride(doneidx, stride, st_delta, stdir, extra)
        up_d, down_d = d1 + s, d2 + maxclr - s + 1 + o2
        total_d = min(up_d, down_d) + std
        if total_d < best_d:
          best_d = total_d
        break
      else:
        if s == maxclr:
          up_d = d2 + 1 + o2
          down_d = up_d + 2
        else:
          up_d = d1 + s + 2
          down_d = d2 + maxclr - s + 1 + o2
      if min(up_d, down_d) < best_d:
        if down_d - up_d > 1:
          off.append(1)
          ds.append(up_d)
        elif up_d >= down_d:
          off.append(-1)
          ds.append(down_d)
        else:      
          off.append(1)
          ds.append(up_d)
          off.append(-1)
          ds.append(down_d)
  else:
    for i, s in enumerate(reversed(station[1:])):
      n = len(off)
      if n == 0:
        break
      o1 = o2 = off.pop(0)
      d1 = d2 = ds.pop(0)
      if n > 1:
        o2 = off.pop(0)
        d2 = ds.pop(0)
      if i == done:
        std = get_dist_ml_to_stride(doneidx, stride, st_delta, stdir, extra)
        up_d, down_d = d1 + s + 1 + o1, d2 + maxclr - s
        total_d = min(up_d, down_d) + std
        if total_d < best_d:
          best_d = total_d
        break
      else:
        if s == maxclr:
          up_d = down_d = d2 + 2
        else:
          up_d = d1 + s + 3 + o1
          down_d = d2 + maxclr - s
      if min(up_d, down_d) < best_d:
        if up_d - down_d > 1:
          off.append(1)
          ds.append(down_d)
        elif down_d >= up_d:
          off.append(-1)
          ds.append(up_d)
        else:      
          off.append(-1)
          ds.append(up_d)
          off.append(1)
          ds.append(down_d)
  return best_d
 
def get_distance_moving_left(station, stride, colors, best_d, stdir='both', doedge=True):
  stnlen = len(station)
  maxidx = stnlen - 1
  maxclr = colors - 1
  if all([s == 0 for s in station]):
    d1 = stride
    d2 = stnlen - stride
    if d1 <= d2:
      return d1 * 2
    else:
      return d2 * 2
  elif all([s == maxclr for s in station]):
    d1 = stride
    d2 = maxidx - stride
    if d1 <= d2:
      return d1 * 2 + 1
    else:
      return d2 * 2 + 1

  doneidx = 1
  s0 = station[0]
  s1 = station[1]
  if s1 in [0, maxclr]:
    for s in station[1:]:
      if s == s1:
        doneidx += 1
      else:
        break
 
  if doneidx > maxidx:
    best_d = get_distance_moving_right(station, stride, colors, best_d)
  else:
    if s1 == 0 and doedge:
      s0_d1 = s0 + 2
      s0_rep1 = s0_d1 - 1
      s0_d2 = maxclr - s0 + 4
      s0_rep2 = 4 - s0_d2
    elif s1 == maxclr and doedge:
      s0_d1 = s0 + 5
      s0_rep1 = s0_d1 - 4
      s0_d2 = maxclr - s0 + 1
      s0_rep2 = 1 - s0_d2
    else:
      s0_d1 = s0 + 2
      s0_rep1 = s0_d1 - 1
      s0_d2 = maxclr - s0 + 1
      s0_rep2 = 1 - s0_d2
 
    rep_off = 0
    if stride == doneidx:
      if s1 in [0, maxclr] and doedge:
        tv = [s1]
        if s0_d1 <= s0_d2:
          ds = [s0_d1]
        else:
          ds = [s0_d2]
      else:
        if abs(s0_d1 - s0_d2) > 0:
          if s0_d1 < s0_d2:
            tv = [0]
            ds = [s0_d1]
          else:
            tv = [maxclr]
            ds = [s0_d2]
        else:
          tv = [0, maxclr]
          ds = [s0_d1, s0_d2]
      for v, d in zip(tv, ds):
        best_d = get_dist_ml_stride_dir(station, stride, 0, 'right', (v, d), maxidx, maxclr, doneidx, best_d)
    else:
      #stride_left
      if stdir in ['both', 'left']:      
        stpos = stride > doneidx
        steq = stride if stpos else (maxidx + 1)
        st_delta = maxidx - steq + 2
        adj_station = []
        rep_off = int(stpos)
        c = 0
        for i, s in enumerate(station):
          if stpos:
            rep = i == doneidx or i == 0 or i > stride
          else:
            rep = i == doneidx
            if i > doneidx and c == 0:
              adj_station.extend(station[i:])
              break
          offset = 1 if rep else 0
          adj_s = s + offset + c
          if adj_s > maxclr:
            adj_s -= colors
            c = 1 if i > 0 else 0
          else:
            c = 0
          adj_station.append(adj_s)
        adj_rep1 = s0_rep1 + rep_off
        abs_rep1 = abs(adj_rep1)
        adj_d1 = s0_d1 + abs_rep1 - abs(s0_rep1)
        adj_rep2 = s0_rep2 + rep_off
        abs_rep2 = abs(adj_rep2)
        adj_d2 = s0_d2 + abs_rep2 - abs(s0_rep2)
        if s1 in [0, maxclr] and doedge:
          tv = [s1]
          if adj_d1 <= adj_d2:
            ds = [adj_d1]
          else:
            ds = [adj_d2]
        else:
          if abs(adj_d1 - adj_d2) > 0:
            if adj_d1 < adj_d2:
              tv = [0]
              ds = [adj_d1]
            else:
              tv = [maxclr]
              ds = [adj_d2]
          else:
            tv = [0, maxclr]
            ds = [adj_d1, adj_d2]
        for v, d in zip(tv, ds):
          best_d = get_dist_ml_stride_dir(adj_station, stride, st_delta, 'left', (v, d), maxidx, maxclr, doneidx, best_d)
    #stride_right
    if stdir in ['both', 'right']:
      if s1 == 0 and not (stride > 0 and stride < doneidx) and doedge:
        s0_d1 = s0 + 2
        s0_rep1 = s0_d1 - 1
        s0_d2 = maxclr - s0 + 4
        s0_rep2 = 4 - s0_d2
      elif s1 == maxclr and not (stride > 0 and stride < doneidx) and doedge:
        s0_d1 = s0 + 5
        s0_rep1 = s0_d1 - 4
        s0_d2 = maxclr - s0 + 1
        s0_rep2 = 1 - s0_d2
      else:
        s0_d1 = s0 + 2
        s0_rep1 = s0_d1 - 1
        s0_d2 = maxclr - s0 + 1
        s0_rep2 = 1 - s0_d2

      stpos = stride > doneidx
      steq = stride if stpos else stnlen
      st_delta = steq - doneidx
      adj_station = []
      c = 0
      rep_off = 0
      if not stpos:
        adj_s0 = s0 - 1
        if adj_s0 < 0:
          adj_s0 += colors
        adj_station.append(adj_s0)
        rep_off = -1
      else:
        adj_station.append(s0)
      for i, s in enumerate(station):
        if i == 0:
          continue
        rep = i > doneidx and i <= steq
        offset = 1 if rep else 0
        adj_s = s - offset - c
        if adj_s < 0:
          adj_s += colors
          c = 1
        else:
          c = 0
        adj_station.append(adj_s)
        if i >= steq and c == 0:
          adj_station.extend(station[i + 1:])
          break
      adj_rep1 = s0_rep1 + rep_off
      abs_rep1 = abs(adj_rep1)
      adj_d1 = s0_d1 + abs_rep1 - abs(s0_rep1)
      adj_rep2 = s0_rep2 + rep_off
      abs_rep2 = abs(adj_rep2)
      adj_d2 = s0_d2 + abs_rep2 - abs(s0_rep2)
      extras = []
      if s1 in [0, maxclr] and not (stride > 0 and stride < doneidx) and doedge:
        extras.append(0)
        tv = [s1]
        if adj_d1 <= adj_d2:
          ds = [adj_d1]
        else:
          ds = [adj_d2]
      else:
        if s1 in [0, maxclr] and (stride > 0 and stride < doneidx):
          if abs(adj_d1 - adj_d2) > 0:
            tv = [s1]
            if adj_d1 < adj_d2:
              ds = [adj_d1]
              if s1 == maxclr:
                extras.append(3)
              else:
                extras.append(0)
            else:
              ds = [adj_d2]
              if s1 == 0:
                extras.append(1)
              else:
                extras.append(0)
          else:
            tv = [s1, s1]
            ds = [adj_d1, adj_d2]
            if s1 == 0:
              extras.extend([0, 1])
            else:
              extras.extend([3, 0])
        else:
          if abs(adj_d1 - adj_d2) > 0:
            extras.append(0)
            if adj_d1 < adj_d2:
              tv = [0]
              ds = [adj_d1]
            else:
              tv = [maxclr]
              ds = [adj_d2]
          else:
            tv = [0, maxclr]
            ds = [adj_d1, adj_d2]
            extras.extend([0, 0])
      for v, d, xt in zip(tv, ds, extras):
        best_d = get_dist_ml_stride_dir(adj_station, stride, st_delta, 'right', (v, d), maxidx, maxclr, doneidx, best_d, xt)
  return best_d
 
def get_windows(station, colors):
  max_idx = len(station) - 1
  max_symbol = colors - 1
  w = False
  windows = []
  winval = 0
  window_start = 0
  for i, d in enumerate(station[1:]):
    if not w and (d == 0 or d == max_symbol):
      window_start = i
      winval = d
      w = True
    elif w and d != winval:
      windows.append([window_start, i + 1, winval])
      if d in [0, max_symbol]:
        window_start = i
        winval = d
        w = True
      else:
        w = False
  if w:
    windows.append([window_start, max_idx + 1, winval])
  return windows
 
def action_distance(station, stride, colors):
  stnlen = len(station)
  maxidx = stnlen - 1
  maxclr = colors - 1
  if all([s == 0 for s in station]):
    d1 = stride
    d2 = stnlen - stride
    if d1 <= d2:
      return d1 * 2
    else:
      return d2 * 2
  elif all([s == maxclr for s in station]):
    d1 = stride
    d2 = maxidx - stride
    if d1 <= d2:
      return d1 * 2 + 1
    else:
      return d2 * 2 + 1
  else:
    #all right or left
    best_d = np.inf
    best_d = get_distance_moving_right(station, stride, colors, best_d)
    best_d = get_distance_moving_left(station, stride, colors, best_d)
    windows = get_windows(station, colors)
    #print(windows)
    for (lowedge, highedge, winval) in windows:
      if lowedge == 0 or highedge == stnlen:
        continue
      #first right then left
      #stride in place
      if stride == highedge:
        adj_station = [s if i <= lowedge else winval for i, s in enumerate(station)]
        dp = get_distance_moving_right(adj_station, maxidx, colors, best_d, stdir='left')
        if dp < best_d:
          best_d = get_dist_ml_stride_dir(station, stride, 0, 'right', (winval, dp), maxidx, maxclr, stride, best_d)
      else:
        #stride right
        if stride > highedge:
          adj_station = [s if i <= lowedge else winval for i, s in enumerate(station)]
          dp = get_distance_moving_right(adj_station, maxidx, colors, best_d, stdir='left')
          if dp < best_d:
            st_delta = stride - highedge
            adj_station = []
            c = 0
            for i, s in enumerate(station):
              if i <= highedge:
                adj_station.append(s)
                continue
              rep = i <= stride
              offset = 1 if rep else 0
              adj_s = s - offset - c
              if adj_s < 0:
                adj_s += colors
                c = 1
              else:
                c = 0
              adj_station.append(adj_s)
              if not rep and c == 0:
                adj_station.extend(station[i + 1:])
                break
            best_d = get_dist_ml_stride_dir(adj_station, stride, st_delta, 'right', (winval, dp), maxidx, maxclr, highedge, best_d)
        else:
          if stride < lowedge:
            steps_forward = stride + 1
            steps_end = 0
            adj_station = []
            c = 0
            for i, s in enumerate(station):
              if i <= stride:
                adj_station.append(s)
                continue
              rep = i <= lowedge
              offset = 1 if rep else 0
              if i <= lowedge:
                adj_s = s + offset + c
              else:
                adj_s = winval + offset + c
              if adj_s > maxclr:
                adj_s -= colors
                c = 1
              else:
                c = 0
              adj_station.append(adj_s)
              if not rep and c == 0:
                adj_station.extend([winval] * (maxidx - i))
                break
          else:
            steps_forward = lowedge + 1
            steps_end = stride - lowedge
            adj_station = [s if i <= lowedge else winval for i, s in enumerate(station)]
          dp = get_distance_moving_right(adj_station, lowedge, colors, best_d)
          if dp < best_d:
            steps_back = lowedge + 1
            dp += steps_back       
            st_delta = maxidx - highedge
            adj_station = []
            c = 0
            for i, s in enumerate(station):
              if i <= highedge:
                adj_station.append(s)
                continue
              adj_s = s - 1 - c
              if adj_s < 0:
                adj_s += colors
                c = 1
              else:
                c = 0
              adj_station.append(adj_s)
            dp = get_dist_ml_stride_dir(adj_station, maxidx, st_delta, 'right', (winval, dp), maxidx, maxclr, highedge, best_d)
            if dp < best_d:
              dp += steps_forward
              dp += steps_end
              if dp < best_d:
                best_d = dp
        #stride left
        if stride >= lowedge and stride < highedge:
          adj_station = [s if i <= lowedge else winval for i, s in enumerate(station)]
          dp = get_distance_moving_right(adj_station, maxidx, colors, best_d, stdir='left')
          if dp < best_d:
            st_delta = 1
            adj_station = []
            c = 0
            for i, s in enumerate(station):
              if i < highedge:
                adj_station.append(s)
                continue
              rep = i == highedge
              offset = 1 if rep else 0
              adj_s = s + offset + c
              if adj_s > maxclr:
                adj_s -= colors
                c = 1
              else:
                c = 0
              adj_station.append(adj_s)
              if not rep and c == 0:
                adj_station.extend(station[i + 1:])
                break
            best_d = get_dist_ml_stride_dir(adj_station, stride, st_delta, 'left', (winval, dp), maxidx, maxclr, highedge, best_d)
        else:
          steq = stride if stride < lowedge else -1
          adj_station = []
          c = 0
          for i, s in enumerate(station):
            if i > lowedge and c == 0:
              adj_station.extend([winval] * (maxidx - i + 1))
              break
            offset = (1 if i <= steq else 2) if i <= lowedge else 0
            if i <= lowedge:
              adj_s = s + offset + c
            else:
              adj_s = winval + offset + c
            if adj_s > maxclr:
              adj_s -= colors
              c = 1
            else:
              c = 0
            adj_station.append(adj_s)
          dp = get_distance_moving_right(adj_station, lowedge, colors, best_d)
          if dp < best_d:
            steps_back = lowedge + 1
            dp += steps_back        
            adj_station = []
            c = 0
            for i, s in enumerate(station):
              if i < highedge:
                adj_station.append(s)
                continue
              rep = i == highedge
              offset = 1 if rep else 0
              adj_s = s + offset + c
              if adj_s > maxclr:
                adj_s -= colors
                c = 1
              else:
                c = 0
              adj_station.append(adj_s)
              if not rep and c == 0:
                adj_station.extend(station[i + 1:])
                break
            steps_end = 0
            if stride > highedge and stride < maxidx:
              steps_end = maxidx - stride
              prev_station = adj_station[:]
              adj_station = []
              c = 0
              for i, s in enumerate(prev_station):
                if i <= stride:
                  adj_station.append(s)
                  continue
                adj_s = s + 1 + c
                if adj_s > maxclr:
                  adj_s -= colors
                  c = 1
                else:
                  c = 0
                adj_station.append(adj_s)
            dp = get_dist_ml_stride_dir(adj_station, highedge, 0, 'left', (winval, dp), maxidx, maxclr, highedge, best_d)
            if dp < best_d:
              dp += 1
              steps_back = highedge - lowedge - 1
              dp += 2 * steps_back
              steps_end += (lowedge - stride) if stride < lowedge else (lowedge + 1)
              dp += steps_end
              if dp < best_d:
                best_d = dp
      #first left then right
      if stride == lowedge:
        doneidx = highedge
        rep_off = -1
        adj_station = []
        c = 0
        for i, s in enumerate([s if i == 0 or i >= highedge else 0 for i, s in enumerate(station)]):
          rep = i > doneidx or i == 0
          offset = 1 if rep else 0
          adj_s = s - offset - c
          if adj_s < 0:
            adj_s += colors
            c = 1
          else:
            c = 0
          adj_station.append(adj_s)
        s0 = adj_station[0]
        s0_d1 = s0 + 2
        s0_rep1 = s0_d1 - 1
        s0_d2 = maxclr - s0 + 1
        s0_rep2 = 1 - s0_d2
        st_delta = stnlen - doneidx
        if abs(s0_d1 - s0_d2) > 0:
          if s0_d1 < s0_d2:
            tv = [0]
            ds = [s0_d1]
          else:
            tv = [maxclr]
            ds = [s0_d2]
        else:
          tv = [0, maxclr]
          ds = [s0_d1, s0_d2]
        prev_station = adj_station[:]
        for v, d in zip(tv, ds):
          carryath = 0
          if (v == 0 or (v == maxclr and s0 == maxclr)):
            if winval == maxclr:
              carryath = 1
          else:
            if winval == 0:
              carryath = -1
          adj_station = []
          c = 0
          for i, s in enumerate(prev_station):
            if i < highedge:
              adj_station.append(s)
              continue
            offset = carryath if i == highedge else 0
            adj_s = s + offset + c
            if adj_s > maxclr:
              adj_s -= colors
              c = 1
            elif adj_s < 0:
              adj_s += colors
              c = -1
            else:
              c = 0
            adj_station.append(adj_s)
          dp = get_dist_ml_stride_dir(adj_station, 0, st_delta, 'right', (v, d), maxidx, maxclr, doneidx, best_d)
          if dp < best_d:
            carryat1 = int(v == maxclr and not (v == maxclr and s0 == maxclr))
            adj_station = []
            c = 0
            for i, s in enumerate(station):
              if i == 0:
                adj_station.append(0)
                continue
              offset = carryat1 if i == 1 else 0
              adj_s = (s if i <= lowedge else winval) + offset + c
              if adj_s > maxclr:
                adj_s -= colors
                c = 1
              else:
                c = 0
              adj_station.append(adj_s)
              if i >= lowedge and c == 0:
                adj_station.extend([winval] * (maxidx - i))
                break
            dp2 = get_distance_moving_right(adj_station, lowedge, colors, best_d, stdir='left')
            dp += dp2
            if dp < best_d:
              best_d = dp
      else:
        #stride right
        doneidx = highedge
        rep_off = -1
        adj_station = []
        c = 0
        for i, s in enumerate([s if i == 0 or i >= highedge else 0 for i, s in enumerate(station)]):
          if stride >= lowedge and stride < highedge: 
            rep = i > doneidx or i == 0
            offset = 1 if rep else 0
          elif stride >= highedge:
            offset = 2 if i > highedge and i <= stride else (1 if i == 0 or i == highedge or i > stride else 0)
          else:
            offset = 2 if i > highedge or i == 0 else (1 if i == highedge else 0)
          adj_s = s - offset - c
          if adj_s < 0:
            adj_s += colors
            c = 1
          else:
            c = 0
          adj_station.append(adj_s)
        s0 = adj_station[0]
        s0_d1 = s0 + 2
        s0_rep1 = s0_d1 - 1
        s0_d2 = maxclr - s0 + 1
        s0_rep2 = 1 - s0_d2
        st_delta = stnlen - doneidx
        if abs(s0_d1 - s0_d2) > 0:
          if s0_d1 < s0_d2:
            tv = [0]
            ds = [s0_d1]
          else:
            tv = [maxclr]
            ds = [s0_d2]
        else:
          tv = [0, maxclr]
          ds = [s0_d1, s0_d2]
        prev_station = adj_station[:]
        for v, d in zip(tv, ds):
          carryath = 0
          if (v == 0 or (v == maxclr and s0 == maxclr)):
            if winval == maxclr:
              carryath = 1
          else:
            if winval == 0:
              carryath = -1
          adj_station = []
          c = 0
          for i, s in enumerate(prev_station):
            if i < highedge:
              adj_station.append(s)
              continue
            offset = carryath if i == highedge else 0
            adj_s = s + offset + c
            if adj_s > maxclr:
              adj_s -= colors
              c = 1
            elif adj_s < 0:
              adj_s += colors
              c = -1
            else:
              c = 0
            adj_station.append(adj_s)
          dp = get_dist_ml_stride_dir(adj_station, 0, st_delta, 'right', (v, d), maxidx, maxclr, doneidx, best_d)
          if dp < best_d:
            carryat1 = int(v == maxclr and not (v == maxclr and s0 == maxclr))
            adj_station = []
            c = 0
            for i, s in enumerate(station):
              if i == 0:
                adj_station.append(0)
                continue
              if stride > 0 and stride <= lowedge:
                if i == 1:
                  offset = (1 if i > 0 and i <= stride else 0) - carryat1
                else:
                  offset = 1 if i > 0 and i <= stride else 0
              else:
                offset = -carryat1 if i == 1 else 0
              adj_s = (s if i <= lowedge else winval) - offset - c
              if adj_s < 0:
                adj_s += colors
                c = 1
              elif adj_s > maxclr:
                adj_s -= colors
                c = -1
              else:
                c = 0
              adj_station.append(adj_s)
            dp2 = get_distance_moving_right(adj_station, lowedge + 1, colors, best_d)
            dp += dp2
            if dp < best_d:
              steps_forward = (stride if stride >= lowedge + 1 and stride < highedge else highedge) - lowedge - 1
              dp += steps_forward * 2
              steps_end = (stride - highedge + 1) if stride >= highedge else ((stride + 1) if stride <= lowedge else 0)
              dp += steps_end
              if dp < best_d:
                best_d = dp
        #stride left
        doneidx = highedge
        rep_off = -1
        adj_station = []
        c = 0
        for i, s in enumerate([s if i == 0 or i >= highedge else 0 for i, s in enumerate(station)]):
          if stride < lowedge: 
            rep = i > doneidx or i == 0
            offset = 1 if rep else 0
          elif stride >= highedge:
            offset = 1 if i > highedge and i <= stride else 0
          else:
            offset = -1 if i == highedge else 0
          adj_s = s - offset - c
          if adj_s < 0:
            adj_s += colors
            c = 1
          elif adj_s > maxclr:
            adj_s -= colors
            c = -1
          else:
            c = 0
          adj_station.append(adj_s)
        s0 = adj_station[0]
        s0_d1 = s0 + 2
        s0_rep1 = s0_d1 - 1
        s0_d2 = maxclr - s0 + 1
        s0_rep2 = 1 - s0_d2
        st_delta = stnlen - doneidx
        if abs(s0_d1 - s0_d2) > 0:
          if s0_d1 < s0_d2:
            tv = [0]
            ds = [s0_d1]
          else:
            tv = [maxclr]
            ds = [s0_d2]
        else:
          tv = [0, maxclr]
          ds = [s0_d1, s0_d2]
        prev_station = adj_station[:]
        for v, d in zip(tv, ds):
          carryath = 0
          if (v == 0 or (v == maxclr and s0 == maxclr and stride < lowedge)):
            if winval == maxclr:
              carryath = 1
          else:
            if winval == 0:
              carryath = -1
          adj_station = []
          c = 0
          for i, s in enumerate(prev_station):
            if i < highedge:
              adj_station.append(s)
              continue
            offset = carryath if i == highedge else 0
            adj_s = s + offset + c
            if adj_s > maxclr:
              adj_s -= colors
              c = 1
            elif adj_s < 0:
              adj_s += colors
              c = -1
            else:
              c = 0
            adj_station.append(adj_s)
          dp = get_dist_ml_stride_dir(adj_station, 0, st_delta, 'right', (v, d), maxidx, maxclr, doneidx, best_d)
          if dp < best_d:
            carryat1 = int(v == maxclr and not (v == maxclr and s0 == maxclr and stride < lowedge))
            adj_station = []
            c = 0
            for i, s in enumerate(station):
              if i == 0:
                adj_station.append(0)
                continue
              if stride < lowedge:
                if i == 1:
                  offset = (1 if i > stride and i <= lowedge else 0) + carryat1
                else:
                  offset = 1 if i > stride and i <= lowedge else 0
              else:
                if i == 1:
                  offset = (1 if i <= lowedge else 0) + carryat1
                else:
                  offset = 1 if i <= lowedge else 0
              adj_s = (s if i <= lowedge else winval) + offset + c
              if adj_s > maxclr:
                adj_s -= colors
                c = 1
              elif adj_s < 0:
                adj_s += colors
                c = -1
              else:
                c = 0
              adj_station.append(adj_s)
            dp2 = get_distance_moving_right(adj_station, lowedge, colors, best_d)
            dp += dp2
            if dp < best_d:
              steps_back = (lowedge - stride) if stride < lowedge else lowedge
              if stride > lowedge:
                steps_back += (maxidx - stride + 1) if stride >= highedge - 1 else (maxidx - highedge + 2)
              dp += steps_back
              steps_end = (highedge - 1 - stride) if stride > lowedge and stride < highedge - 1 else 0
              dp += steps_end
              if dp < best_d:
                best_d = dp
  return best_d
