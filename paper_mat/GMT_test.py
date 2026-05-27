import pygmt

topo = pygmt.datasets.load_earth_relief("10m")

print('shape of topo,', topo.shape)
print('tupe of topo,', type(topo))
print('coordinates of topo,', topo.coords['lat'].shape)


fig = pygmt.Figure()
fig.basemap(region="g", projection="N20c", frame="a")
fig.grdimage(topo, cmap="geo")
fig.colorbar(position="JCR+v",
             frame=["x2000", "y+lm"])
fig.savefig('gmt_test.png',
            show=False, dpi=300)
