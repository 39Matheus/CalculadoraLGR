import streamlit as st
import numpy as np
import sympy as sp
import control as ctrl
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Calculadora LGR Completa", layout="wide")

class AnalisadorLGR:
    def __init__(self, numG, denG, numH, denH):
        self.G = ctrl.TransferFunction(numG, denG)
        self.H = ctrl.TransferFunction(numH, denH)
        self.GH = self.G * self.H
        
        self.s, self.K = sp.symbols('s K')
        self.w = sp.Symbol('w', real=True)
        
        num_arr = self.GH.num[0][0]
        den_arr = self.GH.den[0][0]
        
        self.num_expr = sum(c * self.s**i for i, c in enumerate(reversed(num_arr)))
        self.den_expr = sum(c * self.s**i for i, c in enumerate(reversed(den_arr)))
        
        self.polos = []
        self.zeros = []
        self.np = 0
        self.nz = 0
        self.min_x = -5
        self.max_x = 5
        self.span = 10
        
        self.fig = go.Figure()

    def passo1_equacao_caracteristica(self):
        st.subheader("Passo 1: Equação Característica")
        eq_caracteristica = self.den_expr + self.K * self.num_expr
        st.latex(r"1 + K \cdot P(s) = 0")
        st.latex(f"1 + K \\cdot \\left( \\frac{{{self.num_expr}}}{{{self.den_expr}}} \\right) = 0")
        st.write("Polinômio Característico Expandido:")
        st.latex(f"{sp.collect(sp.expand(eq_caracteristica), self.s)} = 0")

    def passo2_polos_zeros(self):
        st.subheader("Passo 2: Fatorar para encontrar Pólos e Zeros")
        raizes_num = sp.roots(self.num_expr, self.s)
        raizes_den = sp.roots(self.den_expr, self.s)
        
        self.zeros = [complex(z) for z in raizes_num.keys() for _ in range(raizes_num[z])]
        self.polos = [complex(p) for p in raizes_den.keys() for _ in range(raizes_den[p])]
        
        self.nz = len(self.zeros)
        self.np = len(self.polos)
        
        # Calcular bounding box para escala dinâmica do gráfico
        all_reals = [float(np.real(p)) for p in self.polos] + [float(np.real(z)) for z in self.zeros]
        if all_reals:
            self.min_x = min(all_reals)
            self.max_x = max(all_reals)
            self.span = max(self.max_x - self.min_x, 4)
        
        # Configurar proporção fixa (Aspect Ratio 1:1) para evitar assíntotas distorcidas
        self.fig.update_layout(
            xaxis_title='Eixo Real (Re)',
            yaxis_title='Eixo Imaginário (Im)',
            showlegend=True,
            hovermode='closest',
            height=500,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(scaleanchor="x", scaleratio=1) # Trava essencial para visualização geométrica correta
        )
        self.fig.add_hline(y=0, line_width=1.5, line_color="black", opacity=0.5)
        self.fig.add_vline(x=0, line_width=1.5, line_color="black", opacity=0.5)
        
        # Buffer invisível para impedir que o eixo Y colapse no zoom
        self.fig.add_trace(go.Scatter(x=[self.min_x - self.span*0.1, self.max_x + self.span*0.1], 
                                      y=[-self.span*0.5, self.span*0.5], 
                                      mode='markers', marker=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'))

        str_zeros = ", ".join([str(np.round(z, 4)) for z in self.zeros]) if self.zeros else "Nenhum"
        str_polos = ", ".join([str(np.round(p, 4)) for p in self.polos]) if self.polos else "Nenhum"
        
        st.write(f"**Zeros ({self.nz}):** {str_zeros}")
        st.write(f"**Pólos ({self.np}):** {str_polos}")
        
        if self.polos:
            self.fig.add_trace(go.Scatter(x=np.real(self.polos), y=np.imag(self.polos), mode='markers',
                                          marker=dict(symbol='x', size=12, color='red', line=dict(width=2)), name='Pólos'))
        if self.zeros:
            self.fig.add_trace(go.Scatter(x=np.real(self.zeros), y=np.imag(self.zeros), mode='markers',
                                          marker=dict(symbol='circle-open', size=12, color='blue', line=dict(width=2)), name='Zeros'))
        
        st.plotly_chart(self.fig, width='stretch', key='fig_p2')

    def passo3_grafico_inicial(self):
        st.subheader("Passo 3: Assinalar pólos (x) e zeros (o)")
        st.info("Pólos marcados em vermelho (X) e Zeros marcados em azul (O) no gráfico acima.")

    def passo4_segmentos_eixo_real(self):
        st.subheader("Passo 4: Segmentos do Eixo Real no LGR")
        pontos_reais = sorted([p.real for p in self.polos if p.imag == 0] + 
                              [z.real for z in self.zeros if z.imag == 0])
        
        limites = [-np.inf] + pontos_reais + [np.inf]
        segmentos = []
        
        for esq, dir_ in zip(limites[:-1], limites[1:]):
            ponto_teste = (esq + dir_)/2 if not np.isinf(esq) and not np.isinf(dir_) else (dir_ - 1 if np.isinf(esq) else esq + 1)
            elementos_a_direita = sum(1 for p in pontos_reais if p > ponto_teste)
            if elementos_a_direita % 2 == 1:
                segmentos.append((esq, dir_))
        
        if segmentos:
            for i, seg in enumerate(segmentos):
                st.write(f"O segmento entre **{seg[0]:.4f}** e **{seg[1]:.4f}** pertence ao LGR.")
                x_start = seg[0] if not np.isinf(seg[0]) else seg[1] - self.span
                x_end = seg[1] if not np.isinf(seg[1]) else seg[0] + self.span
                self.fig.add_trace(go.Scatter(x=[x_start, x_end], y=[0, 0], mode='lines',
                                              line=dict(color='green', width=4), name=f'Segmento Real {i+1}'))
        else:
            st.write("Nenhum segmento no eixo real pertence ao LGR.")
            
        st.plotly_chart(self.fig, width='stretch', key='fig_p4')

    def passo5_lugares_separados(self):
        st.subheader("Passo 5: Lugares Separados (Ramos)")
        ls = max(self.np, self.nz)
        st.write(f"Número de curvas (Ramos) = $\\max(N_p, N_z) = {ls}$")

    def passo6_simetria(self):
        st.subheader("Passo 6: Simetria")
        st.write("O Lugar Geométrico das Raízes é simétrico em relação ao eixo real.")

    def passo7_assintotas(self):
        st.subheader("Passo 7: Assíntotas")
        num_assintotas = self.np - self.nz
        if num_assintotas > 0:
            sigma_A = (sum(self.polos) - sum(self.zeros)) / num_assintotas
            sigma_A = np.real(sigma_A)
            angulos_A = [((2*q + 1) * 180) / num_assintotas for q in range(num_assintotas)]
            
            st.write(f"O número de assíntotas é $N_p - N_z = {num_assintotas}$")
            st.write(f"**Centro das assíntotas ($\\sigma_A$):** {sigma_A:.4f}")
            st.write(f"**Ângulos das assíntotas:** {', '.join([f'{a}°' for a in angulos_A])}")
            
            raio = self.span * 1.5
            for i, ang in enumerate(angulos_A):
                rad = np.radians(ang)
                dx = np.cos(rad)
                dy = np.sin(rad)
                
                # Previne ruído de ponto flutuante esticando o eixo Y erroneamente
                if abs(dx) < 1e-10: dx = 0.0
                if abs(dy) < 1e-10: dy = 0.0
                
                x_end = sigma_A + raio * dx
                y_end = raio * dy
                
                self.fig.add_trace(go.Scatter(x=[sigma_A, x_end], y=[0, y_end], mode='lines',
                                              line=dict(color='gray', width=1.5, dash='dash'), name=f'Assíntota {i+1}'))
        else:
            st.write("Não há assíntotas tendendo ao infinito ($N_p \\le N_z$).")
            
        st.plotly_chart(self.fig, width='stretch', key='fig_p7')

    def passo8_pontos_saida_chegada(self):
        st.subheader("Passo 8: Pontos de Saída/Chegada no Eixo Real")
        den_diff = sp.diff(self.den_expr, self.s)
        num_diff = sp.diff(self.num_expr, self.s)
        eq_pontos = den_diff * self.num_expr - self.den_expr * num_diff
        
        st.write("Solucionando $\\frac{dK}{ds} = 0$:")
        raizes_eq = sp.roots(eq_pontos, self.s)
        pontos = [complex(r) for r in raizes_eq.keys()]
        
        if pontos:
            for p in pontos:
                try:
                    num_val = complex(self.num_expr.subs(self.s, p))
                    den_val = complex(self.den_expr.subs(self.s, p))
                    K_val = -den_val / num_val if num_val != 0 else float('inf')
                    
                    if abs(p.imag) < 1e-5 and K_val.real > 0:
                        st.success(f"Ponto VÁLIDO no eixo real: $s = {p.real:.4f}$ (com $K = {K_val.real:.4f}$)")
                        self.fig.add_trace(go.Scatter(x=[p.real], y=[0], mode='markers',
                                                      marker=dict(symbol='square', size=8, color='purple'), name='Saída/Chegada'))
                    else:
                        st.write(f"Ponto ignorado: $s = {np.round(p,4)}$ ( $K = {np.round(K_val,4)}$ - complexo ou $K \\le 0$)")
                except:
                    st.write(f"Ponto $s = {np.round(p,4)}$ (Erro ao avaliar K)")
        else:
            st.write("Não há pontos de saída/chegada solucionáveis analiticamente nesta equação.")

    def passo9_cruzamento_eixo_imaginario(self):
        st.subheader("Passo 9: Cruzamento do Eixo Imaginário")
        den_jw = sp.expand(self.den_expr.subs(self.s, sp.I * self.w))
        num_jw = sp.expand(self.num_expr.subs(self.s, sp.I * self.w))
        
        re_den, im_den = sp.re(den_jw), sp.im(den_jw)
        re_num, im_num = sp.re(num_jw), sp.im(num_jw)
        
        eq_w = sp.simplify(im_den * re_num - re_den * im_num)
        
        try:
            poly_w = sp.Poly(eq_w, self.w)
            coeffs_w = [float(c) for c in poly_w.all_coeffs()]
            raizes_w = np.roots(coeffs_w)
            
            cruzamentos = []
            for w_val in raizes_w:
                if np.isclose(w_val.imag, 0, atol=1e-5): 
                    w_real = float(w_val.real)
                    num_val = float(re_num.subs(self.w, w_real))
                    
                    if abs(num_val) > 1e-5:
                        k_val = -float(re_den.subs(self.w, w_real)) / num_val
                    else:
                        num_im_val = float(im_num.subs(self.w, w_real))
                        if abs(num_im_val) > 1e-5:
                            k_val = -float(im_den.subs(self.w, w_real)) / num_im_val
                        else:
                            k_val = -1
                            
                    if k_val > 0:
                        cruzamentos.append((w_real, k_val))
            
            if cruzamentos:
                for w, k in set(cruzamentos):
                    st.write(f"Cruza o eixo imaginário em **$s = \\pm {w:.4f}j$** para **$K = {k:.4f}$**")
                    self.fig.add_trace(go.Scatter(x=[0, 0], y=[w, -w], mode='markers',
                                                  marker=dict(symbol='diamond', size=8, color='orange'), name='Cruzamento Im'))
            else:
                st.write("O LGR não cruza o eixo imaginário para $K > 0$.")
        except Exception:
            st.warning(f"O método analítico não encontrou cruzamentos (Equação sem raízes reais puras).")

        st.plotly_chart(self.fig, width='stretch', key='fig_p9')

    def passo10_angulos_partida_chegada(self):
        st.subheader("Passo 10: Ângulos de Partida e Chegada")
        def calc_angulo(ponto_alvo, is_polo=True):
            soma_polos = sum(np.angle(ponto_alvo - p) for p in self.polos if np.abs(ponto_alvo - p) > 1e-5)
            soma_zeros = sum(np.angle(ponto_alvo - z) for z in self.zeros if np.abs(ponto_alvo - z) > 1e-5)
            if is_polo:
                ang_rad = np.pi - soma_polos + soma_zeros
            else:
                ang_rad = np.pi + soma_polos - soma_zeros
            return (np.degrees(ang_rad) + 360) % 360

        polos_complexos = [p for p in self.polos if p.imag != 0]
        zeros_complexos = [z for z in self.zeros if z.imag != 0]
        
        if not polos_complexos and not zeros_complexos:
            st.write("Não há pólos nem zeros complexos no sistema.")
            return

        if polos_complexos:
            for p in set(polos_complexos):
                st.write(f"Ângulo de **partida** do pólo $s = {np.round(p,4)}$: **{calc_angulo(p, True):.2f}°**")
                
        if zeros_complexos:
            for z in set(zeros_complexos):
                st.write(f"Ângulo de **chegada** do zero $s = {np.round(z,4)}$: **{calc_angulo(z, False):.2f}°**")

    def passo11_teste_fase(self, s_teste):
        st.subheader(f"Passo 11: Teste do Ângulo de Fase para $s = {s_teste}$")
        soma_angulos_zeros = sum(np.angle(s_teste - z) for z in self.zeros)
        soma_angulos_polos = sum(np.angle(s_teste - p) for p in self.polos)
        
        fase_rad = soma_angulos_zeros - soma_angulos_polos
        fase_deg = np.degrees(fase_rad) % 360
        
        st.write(f"Fase(P(s)) em $s_{{teste}} = {fase_deg:.2f}^\\circ$")
        
        # Correção: LGR (K>0) exige estritamente 180° (módulo 360). 0° é só para K<0.
        if np.isclose(fase_deg, 180, atol=1e-2): 
            st.success("O ponto **PERTENCE** ao LGR (satisfaz condição de ângulo para $K > 0$).")
            return True
        else:
            st.error("O ponto **NÃO PERTENCE** ao LGR (condição de ângulo não satisfeita para $K > 0$).")
            return False

    def passo12_calculo_k(self, s_teste):
        st.subheader(f"Passo 12: Cálculo do Ganho $K$ e Plot do Ponto")
        
        # Injeta o ponto visualmente no gráfico
        self.fig.add_trace(go.Scatter(x=[s_teste.real], y=[s_teste.imag], mode='markers',
                                      marker=dict(symbol='star', size=14, color='magenta', line=dict(width=1, color='black')), 
                                      name='Ponto Teste'))
        
        if self.passo11_teste_fase(s_teste):
            mod_zeros = np.prod([np.abs(s_teste - z) for z in self.zeros]) if self.zeros else 1.0
            mod_polos = np.prod([np.abs(s_teste - p) for p in self.polos]) if self.polos else 1.0
            K_calculado = mod_polos / mod_zeros
            st.info(f"O ganho $K$ requerido para que $s = {s_teste}$ seja raiz é **$K = {K_calculado:.4f}$**")
        else:
            st.write("O cálculo de $K$ é ignorado pois o ponto não pertence ao LGR.")
            
        # Atualiza a figura com o ponto desenhado
        st.plotly_chart(self.fig, width='stretch', key='fig_p12')

    def plotar_lgr_plotly(self):
        st.subheader("Gráfico Final (Traçado Exato)")
        rlist, klist = ctrl.root_locus(self.GH, plot=False)

        for i in range(rlist.shape[1]):
            self.fig.add_trace(go.Scatter(
                x=np.real(rlist[:, i]),
                y=np.imag(rlist[:, i]),
                mode='lines',
                line=dict(width=2.5),
                name=f'Ramo Final {i+1}'
            ))

        st.plotly_chart(self.fig, width='stretch', key='fig_final')


# ==========================================
# INTERFACE STREAMLIT
# ==========================================
st.title("Lugar Geométrico das Raízes: Análise 12 Passos")
st.markdown("Insira os coeficientes dos polinômios separados por vírgulas, partindo do coeficiente de maior grau para o menor.")

with st.form("lgr_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Planta $G(s)$**")
        # Defaults alterados para espelhar K(s+2) / (s^2 + 4s) do PDF
        num_g_input = st.text_input("Numerador $G(s)$", "1, 2")
        den_g_input = st.text_input("Denominador $G(s)$", "1, 4, 0")
        
    with col2:
        st.markdown("**Sensor $H(s)$**")
        num_h_input = st.text_input("Numerador $H(s)$", "1")
        den_h_input = st.text_input("Denominador $H(s)$", "1")
        
    st.markdown("---")
    st.markdown("**Testar se um ponto pertence ao LGR (Passos 11 e 12)**")
    col_pt1, col_pt2 = st.columns(2)
    with col_pt1:
        teste_real = st.number_input("Parte Real do Ponto ($Re$)", value=-2.4, format="%0.4f")
    with col_pt2:
        teste_imag = st.number_input("Parte Imaginária do Ponto ($Im$)", value=0.0000, format="%0.4f")
        
    submit_button = st.form_submit_button(label="Calcular e Gerar Relatório")

if submit_button:
    try:
        num_G = [float(x.strip()) for x in num_g_input.split(',')]
        den_G = [float(x.strip()) for x in den_g_input.split(',')]
        num_H = [float(x.strip()) for x in num_h_input.split(',')]
        den_H = [float(x.strip()) for x in den_h_input.split(',')]
        
        ponto_teste = complex(teste_real, teste_imag)
        analisador = AnalisadorLGR(num_G, den_G, num_H, den_H)
        
        st.markdown("---")
        st.header("Relatório Passo-a-Passo")
        
        analisador.passo1_equacao_caracteristica()
        st.divider()
        analisador.passo2_polos_zeros()
        st.divider()
        analisador.passo3_grafico_inicial()
        st.divider()
        analisador.passo4_segmentos_eixo_real()
        st.divider()
        analisador.passo5_lugares_separados()
        st.divider()
        analisador.passo6_simetria()
        st.divider()
        analisador.passo7_assintotas()
        st.divider()
        analisador.passo8_pontos_saida_chegada()
        st.divider()
        analisador.passo9_cruzamento_eixo_imaginario()
        st.divider()
        analisador.passo10_angulos_partida_chegada()
        st.divider()
        analisador.passo12_calculo_k(ponto_teste)
        st.divider()
        analisador.plotar_lgr_plotly()
        
    except Exception as e:
        st.error(f"Erro na execução! Verifique os dados inseridos. Detalhes do erro: {e}")