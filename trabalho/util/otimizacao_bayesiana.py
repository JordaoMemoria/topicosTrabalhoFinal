import numpy as np


class Dominio():
    def __init__(self, dominio):
        self._dominio = dominio
    
    @property
    def dominio(self):
        dominio_tratado = []
        
        for d in self._dominio:
            if d['type'] == 'categorical':
                dominio_tratado.append(self.tratar_dominio_categorico(d))

            elif d['type'] == 'array':
                dominio_tratado += self.tratar_dominio_array(d)

            else:
                dominio_tratado.append(dict(d))

        return dominio_tratado

    def tratar_dominio_categorico(self, dominio_original):
        tratado = dict(dominio_original)
        tratado['domain'] = list(range(len(dominio_original['domain'])))
        return tratado

    def tratar_dominio_array(self, dominio_original):
        dominios_tratados = []
        for i in range(dominio_original['size']):
            tratado = dict(dominio_original)

            tratado['name'] = f"{tratado['name']}_{i}"
            del tratado['size']
            tratado['type'] = 'continuous'

            dominios_tratados.append(tratado)

        return dominios_tratados
            
    def gerar_f(self, f, **kwargs):
        def funcao(x):
            valores = dict(kwargs)
            
            i = 0
            for d_i in self._dominio:
                if d_i['type'] == 'array':
                    x_ = x[0][i: i+d_i['size']]
                    valor = x_
                    i = i+d_i['size']

                elif d_i['type'] == 'categorical':
                    x_i = x[0][i]
                    valor = d_i['domain'][int(x_i)]
                    i += 1

                else:
                    valor = x[0][i]
                    i += 1
                
                valores[d_i['name']] = valor

            return f(**valores)

        return funcao
