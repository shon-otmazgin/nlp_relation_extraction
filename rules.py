def rule_retired(person, org, sent):
    tokens = sent.split()
    if 'retired' in tokens:
        p_idx = tokens.index(person.split()[-1])
        o_idx = tokens.index(org.split()[-1])
        r_idx = tokens.index('retired')
        if (p_idx < r_idx < o_idx) or (o_idx < r_idx < p_idx):
          if r_idx + 2 >= o_idx:
            return True
    return False


def rule_org_s(person, org, sent):
    with open('data/lexicon.location', 'r', encoding='utf8') as f:
        lex_loc = set([loc.strip() for loc in f])
    tokens = sent.split()
    if "'s" in tokens:
        p_idx = tokens.index(person.split()[-1])
        o_idx = tokens.index(org.split()[-1])
        s_idx = tokens.index("'s")
        if o_idx < s_idx < p_idx:
            if s_idx - 1 == o_idx:
                if org in lex_loc:
                    return True
    return False
