from snia_rise.simulate.mock_lc import RedbackLightCurveLib

RedbackLightCurveLib.simulate_mock_light_curve(n_lc=500, model="curved_power_law")
RedbackLightCurveLib.simulate_mock_light_curve(n_lc=500, model="power_law")
