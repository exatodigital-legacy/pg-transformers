"""Self-contained reference corpus for parity testing.

Original sentences written for this project (redistributable), covering the
languages and phenomena the tokenizers must get right: Portuguese, English,
French, accents, legal-style citations, numbers and currency, Unicode edge
cases, and documents long enough to exercise every position embedding up to
each model's max_tokens. Deterministic: same list every run.
"""

SENTENCES = [
    # Portuguese (accents, cedilla, legal register)
    "A sentença reconheceu a rescisão indireta do contrato de trabalho.",
    "O tribunal negou provimento ao recurso e manteve a decisão de primeira instância.",
    "As partes celebraram acordo homologado em audiência de conciliação.",
    "A petição inicial foi indeferida por inépcia, nos termos da legislação processual.",
    "O réu foi citado por edital após esgotadas as tentativas de localização.",
    "A empresa apresentou contestação alegando prescrição da pretensão autoral.",
    "Ficou comprovado o vínculo empregatício no período de março a novembro.",
    "O perito concluiu que a assinatura aposta no documento era autêntica.",
    "A liminar foi deferida para suspender os efeitos do ato administrativo.",
    "Não há óbice à homologação do plano de recuperação judicial da companhia.",
    "O magistrado determinou a expedição de ofício ao órgão previdenciário.",
    "A testemunha confirmou que o expediente encerrava às dezoito horas.",
    "São devidos honorários advocatícios de dez por cento sobre o valor da causa.",
    "A execução fiscal foi suspensa em razão do parcelamento do débito tributário.",
    "O condomínio ajuizou ação de cobrança das cotas condominiais em atraso.",
    # English
    "The court granted the motion to dismiss for lack of jurisdiction.",
    "Both parties agreed to settle the dispute through binding arbitration.",
    "The witness testified that the shipment arrived two weeks late.",
    "Summary judgment was denied because material facts remained in dispute.",
    "The contract includes a confidentiality clause and a non-compete covenant.",
    # French (apostrophes, accents, guillemets)
    "Le tribunal a rejeté la demande pour défaut de fondement juridique.",
    "L'huissier a signifié l'assignation au défendeur à son domicile.",
    "La cour d'appel a confirmé le jugement rendu en première instance.",
    "Les parties s'engagent à exécuter l'accord « de bonne foi ».",
    "Le préjudice a été évalué à trois mille euros par l'expert désigné.",
    # citations, numbers, currency, dates
    "Processo nº 0001234-56.2024.5.02.0999, valor da causa R$ 1.234.567,89.",
    "Conforme o art. 483, §2º, da CLT, c/c o art. 5º, XXXV, da CF/88.",
    "Invoice #2024-118 totals USD 45,300.00, due on 2024-12-31.",
    "L'article 1219 du Code civil prévoit l'exception d'inexécution.",
    "Audiência designada para 12/05/2025 às 14h30 na 3ª Vara do Trabalho.",
    # mixed and quirky
    "The naïve café owner filed a déjà-vu complaint, à propos of nothing.",
    "E-mail: contato@exemplo.com.br; URL: https://exemplo.com/a?b=1&c=2.",
    "Half ½ ligature ﬁle ﬂow №5 Ⅷ and full-width ｔｅｘｔ１２３.",
    "Emoji no meio 😀 do texto e bandeira 🇧🇷 do Brasil.",
    "Hífen-composto, auto-aplicável, ex-empregado: palavras compostas.",
]

EDGE_STRINGS = [
    "a",
    "UPPER lower MiXeD",
    "  espaços   múltiplos  e\ttabs\n",
    "quotes “curly” and ‘single’ and «guillemets»",
    "dash — em, – en, − minus",
    "中文法律文件 混合 português 텍스트",
    "日本語のテキストです",
    "word" + "x" * 120 + "end",
    "supercalifragilisticexpialidocious" * 4,
    "MAIÚSCULAS ACENTUADAS: ÁÉÍÓÚ ÂÊÔ ÃÕ Ç",
    "Ação:rescisão(art.483,CLT).Sem espaços!",
]


def build():
    """~120 texts: singles, pairs, and long concatenations (>512 tokens)."""
    texts = list(SENTENCES) + list(EDGE_STRINGS)
    # pairs -> medium length
    for i in range(0, len(SENTENCES) - 1, 2):
        texts.append(SENTENCES[i] + " " + SENTENCES[i + 1])
    # rolling concatenations -> long documents (over any model's max_tokens)
    for start in range(0, len(SENTENCES), 3):
        chunk = [SENTENCES[(start + j) % len(SENTENCES)] for j in range(30)]
        texts.append(" ".join(chunk))
    # very long: full corpus twice (truncation path)
    texts.append(" ".join(SENTENCES * 2))
    return [t for t in texts if t.strip()]
