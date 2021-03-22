class Dominio():
    def __init__(self, dominio):
        self._dominio = dominio
    
    @property
    def dominio(self):
        dominio_tratado = []
        
        for d in self._dominio:
            d_tratado = dict(d)
            if d['type'] == 'categorical':
                d_tratado['domain'] = list(range(len(d['domain'])))

            dominio_tratado.append(d_tratado)

        return dominio_tratado
    
    def gerar_f(self, f, **kwargs):
        def funcao(x):
            valores = dict(kwargs)
            
            for x_i, d_i in zip(x[0], self._dominio):
                valor = d_i['domain'][int(x_i)] if d_i['type'] == 'categorical' else x_i
                
                valores[d_i['name']] = valor

            return f(**valores)

        return funcao