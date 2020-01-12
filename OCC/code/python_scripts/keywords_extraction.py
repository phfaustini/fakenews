# python dataset_path language
# language = "datasets/WhatsApp/whats_br/Structured/"

from sys import argv

import RAKE
import stop_words

dataset = argv[1]
lang = argv[2]
"""
dataset_prefix = argv[2]

lang = dataset_prefix.split("/")[2]
if lang == "whats_br" or lang == "FakeBrCorpus" of lang == "tweets_br"
    lang = 'pt'
elif lang == "btv-lifestyle"
    lang = 'bg'
else:
    lang = 'en'
"""
text = '''
Tradicionalmente, desde 1949, cabe ao representante do Brasil abrir o debate geral da assembleia das Nações Unidas. Foi o primeiro pronunciamento de Bolsonaro como chefe de Estado no encontro.

"É uma falácia dizer que a Amazônia é um patrimônio da humanidade e um equívoco, como atestam os cientistas, afirmar que a Amazônia, a nossa floresta, é o pulmão do mundo. Valendo-se dessas falácias um ou outro país, em vez de ajudar, embarcou nas mentiras da mídia e se portou de forma desrespeitosa e com espírito colonialista. Questionaram aquilo que nos é mais sagrado, a nossa soberania", disse Bolsonaro.
Bolsonaro afirmou, ainda, que tem "compromisso solene" com a proteção da Amazônia. Disse que a Amazônia é maior do que toda a Europa ocidental e "permanece praticamente intocada", o que seria prova, segundo o presidente, de que o Brasil é "um dos países que mais protegem o meio ambiente".


"Em primeiro lugar, meu governo tem o compromisso solene com a preservação do meio ambiente e do desenvolvimento sustentável em benefício do Brasil", declarou o presidente.

O discurso do presidente tem o contexto da crise provocada, em agosto, pela alta das queimadas na floresta amazônica.

Bolsonaro trocou farpas com o presidente da França, Emmanuel Macron, que deixou em aberto a discussão sobre um possível status internacional na Amazônia.

Com a fala desta terça, Bolsonaro é o oitavo presidente brasileiro a abrir os debates. O primeiro chefe de Estado do país a discursar no encontro foi João Figueiredo, em 1982. Desde então, apenas Itamar Franco não se pronunciou ao menos uma vez na assembleia geral.

O que revelam os discursos de presidentes brasileiros na ONU
Bolsonaro fala sobre soberania e o espírito colonialista de outros países

'Mentalidade colonialista'
O presidente afirmou na ONU que seu governo tem política de "tolerância zero" com a criminalidade, o que inclui crime ambientais. Bolsonaro ressaltou que a Amazônia não é consumida pelo fogo.

"Ela [Amazônia] não está sendo devastada e nem consumida pelo fogo, como diz mentirosamente a mídia. Cada um de vocês pode comprovar o que estou falando agora", declarou.

Segundo Bolsonaro, o Brasil tem 61% do território preservado e utiliza 8% das terras para produzir alimentos, enquanto França e Alemanha, conforme ele, usam 50% de suas terras.


Bolsonaro ainda afirmou que a ONU "não pode aceitar" o retorno do que considera uma "mentalidade" colonialista e declarou que iniciativas de ajuda ou apoio à preservação da floresta e de outros biomas deverão respeitas a soberania brasileira.

"Quero reafirmar minha posição de que qualquer iniciativa de ajuda ou apoio à preservação da floresta amazônica, ou de outros biomas, deve ser tratado em pleno respeito à soberania brasileira. Estamos prontos para, em parcerias e agregando valor, aproveitar de forma sustentável todo o nosso potencial", disse.
'''

#with (open(dataset, 'r')) as f:
#    text = f.read()

r = RAKE.Rake(stop_words.get_stop_words(lang))

r.run(text)
