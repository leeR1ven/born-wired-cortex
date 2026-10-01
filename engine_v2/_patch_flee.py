*** Begin Patch
*** Update File: tools/chase_task.py
@@
-def _chase_row(got, bearing, fell=None):
+def _chase_row(got, bearing, flee=CHASE_FLEE, fell=None):
@@
-    return dict(bearing=bearing, distance=CHASE_DISTANCE, flee=CHASE_FLEE,
+    return dict(bearing=bearing, distance=CHASE_DISTANCE, flee=flee,
@@
-        row = _chase_row(got, bearing)
+        row = _chase_row(got, bearing, flee)
*** End Patch
