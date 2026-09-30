from pathlib import Path
import mph

folder = Path('Simulations/trap_capacitance').resolve()
client = mph.start(version='6.1', cores=4)
for suffix in ('setup', 'meshed'):
    path = folder / f'COMSOL_Zhi75um_oxide5um_RF37_1V_{suffix}.mph'
    model = client.load(str(path))
    geom = model.java.component('comp1').geom('geom1')
    geom.feature('buried').label('Patterned GND 8/0 | z = -5.5 to -5 um')
    geom.feature('refill').label('SiO2 in GND openings | z = -5.5 to -5 um')
    selections = model.java.component('comp1').selection()
    for tag in selections.tags():
        selection = selections(tag)
        label = str(selection.label())
        if '-5.5 to -3 um' in label:
            selection.label(label.replace('-5.5 to -3 um', '-5.5 to -5 um'))
    model.save(str(path))
    print('Saved:', path.name, flush=True)
    client.remove(model)
