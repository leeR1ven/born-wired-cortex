import io
p = r"F:\born-wired-cortex\engine_v2\tools\probe_gaze_walking.py"
s = io.open(p, encoding="utf-8", newline="").read()
s = s.replace("""            rates = brain.network.rates
            turn = ""
            if chase is not None:
                turn = "%.2f/%.2f" % (float(np.mean(rates[brain.groups["chase_turn_left"]])),
                                      float(np.mean(rates[brain.groups["chase_turn_right"]])))""",
"""            left_rate = float(np.mean(brain.network.rates_at(yaw_l)))
            right_rate = float(np.mean(brain.network.rates_at(yaw_r)))
            turn = ""
            if chase is not None:
                turn = "%.2f/%.2f" % (
                    float(np.mean(brain.network.rates_at(brain.groups["chase_turn_left"]))),
                    float(np.mean(brain.network.rates_at(brain.groups["chase_turn_right"]))))""")
s = s.replace("""                     float(np.mean(rates[yaw_l])), float(np.mean(rates[yaw_r])), turn))""",
              """                     left_rate, right_rate, turn))""")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")