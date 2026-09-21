from pathlib import Path
import sys
import time
import nbformat
import numpy as np

folder = Path('Simulations/trap_capacitance')
original = folder / 'COMSOL_import_GDS_patterned_buried_GND.ipynb'
target = folder / 'COMSOL_import_GDS_patterned_buried_GND_oxide5um.ipynb'

if '--create' in sys.argv:
    nb = nbformat.read(original, 4)
    replacements = [
        ('oxide3um', 'oxide5um'),
        ('OXIDE_ABOVE_GND = 3.0', 'OXIDE_ABOVE_GND = 5.0'),
        ('3 um oxide', '5 um oxide'),
        ('upper 3 um', 'upper 5 um; six sweep layers (at least five)'),
        ('−690.5', '−692.5'), ('−15.5', '−17.5'),
        ('−3.5', '−5.5'), ('−3', '−5'),
        ('-690.5', '-692.5'), ('-15.5', '-17.5'),
        ('-3.5', '-5.5'), ('-3 to', '-5 to'),
        ('3.5 µm of oxide', '5.5 µm of oxide'),
        ('3 µm of oxide', '5 µm of oxide'),
        ('through 3 µm', 'through 5 µm in six layers'),
    ]
    for cell in nb.cells:
        for before, after in replacements:
            cell.source = cell.source.replace(before, after)
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            compile(cell.source, str(target), 'exec')
    nb.cells[0].source = nb.cells[0].source.replace(
        '# GDS trap with a patterned buried ground',
        '# GDS trap with a patterned buried ground — 5 µm spacer')
    nb.cells[0].source = nb.cells[0].source.replace(
        '**You choose the RF domain.**',
        '**RF domain 37 is held at 1 V; all other electrodes are at 0 V.**')
    nbformat.validate(nb)
    with target.open('x', encoding='utf-8') as stream:
        nbformat.write(nb, stream)
    print('Created:', target)
    sys.exit()

nb = nbformat.read(target, 4)
ns = {'__name__': '__main__'}
started = time.monotonic()
for index, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        print(f'CELL {index} ({time.monotonic()-started:.1f}s)', flush=True)
        exec(compile(cell.source, f'{target.name}:cell{index}', 'exec'), ns)

mesh, geom, comp = ns['mesh'], ns['geom'], ns['comp']
assert mesh.feature('sweep_upper').feature('dist1').getInt('numelem') >= 5
assert not mesh.isAutomatic()
for tag, limits in {'top': (0, .5), 'buried': (-5.5, -5),
                    'oxide_inside': (-5, 0), 'oxide_outside': (-5, 0),
                    'refill': (-5.5, -5), 'lower_oxide': (-17.5, -5.5),
                    'silicon': (-692.5, -17.5)}.items():
    for domain in ns['domains'](tag):
        assert np.allclose(ns['bounding_box'](3, domain)[4:], limits)
volume_domains, surface_boundaries = set(), set()
for kind in mesh.getTypes():
    kind = str(kind)
    if kind in ('tet', 'prism', 'pyr', 'hex'):
        volume_domains.update(int(i) for i in np.unique(np.asarray(mesh.getElemEntity(kind))))
    elif kind in ('tri', 'quad'):
        surface_boundaries.update(int(i) for i in np.unique(np.asarray(mesh.getElemEntity(kind))))
assert volume_domains == set(range(1, int(geom.getNDomains())+1))
assert surface_boundaries == set(range(1, int(geom.getNBoundaries())+1))
rf = set(int(i) for i in comp.selection('RF').entities())
ground = set(int(i) for i in comp.selection('GND').entities())
assert rf == set(ns['boundaries']([37]))
assert ground == set(ns['boundaries']([i for i in ns['top_ids'] if i != 37] + ns['buried_ids']))
assert not rf.intersection(ground)
assert np.isclose(float(ns['j'].param().evaluate('Vrf')), 1)
print(f'VERIFIED: 5 um spacer, 6 sweep layers, RF 37 = 1 V, other electrodes grounded; '
      f'{len(volume_domains)} domains and {len(surface_boundaries)} boundaries meshed. '
      f'Elapsed: {time.monotonic()-started:.1f}s.', flush=True)
