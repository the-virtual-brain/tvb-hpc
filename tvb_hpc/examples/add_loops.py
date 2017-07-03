import loopy as lp

kernel = """
<> dx = a * x + b * y
<> dy = c * x + d * y
xn = x + dt * dx {nosync=*}
yn = y + dt * dy {nosync=*}
"""

print('kernel instructions')
print(kernel)

pars = 'a b c d'.split()
state = 'x y xn yn'.split()

knl = lp.make_kernel("{:}", kernel)
knl = lp.add_and_infer_dtypes(knl, {'a,b,c,d,x,y,dt,xn,yn': 'f'})

print('\nwithout loops')
print(lp.generate_code(knl)[0])

print('\ntime stepping')
knl = lp.to_batched(knl, 'nt', state, 'it')
print(lp.generate_code(knl)[0])

print('\nvary param a')
knl = lp.to_batched(knl, 'na', state + ['a'], 'ia')
print(lp.generate_code(knl)[0])

print('\nvary param b')
knl = lp.to_batched(knl, 'nb', state + ['b'], 'ib')
print(lp.generate_code(knl)[0])

print('\nmap ia & ib to global and local dims')
print('notice how ia & ib replaced by get_local_id & get_global_id')
knl = lp.tag_inames(knl, [('ia', 'g.0'), ('ib', 'l.0')], force=True)
print(lp.generate_code(knl)[0])
