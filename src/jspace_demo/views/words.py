"""The words a readout surfaces, as the screens show them: strength bands, concepts and concern alerts."""

from __future__ import annotations

from jspace_demo.glossary import Glossary, singular

from collections.abc import Iterable

STRONG, WEAK = 10, 100  # 1-based rank bands: strong within the top 10, weak within the top 100

# Words that say nothing about what the model is thinking of (function words, auxiliaries, filler). Readouts at
# a pronoun or a verb often predict the next word ("I" -> want), which is not a concept worth showing.
STOPWORDS = frozenset(
    """
a an the this that these those it its i me my mine you your yours we us our they them their he him his she her
myself yourself to of in on at for with from by as about into over after before up down out off and or but so
if than then because while is am are was were be been being do does did done have has had will would shall
should may might must can could cannot not no yes ok okay yeah please thanks thank hi hello what which who whom
whose why how when where whats there here very too just really also even only still yet again now maybe pretty
anyway anymore lately recently actually want wants wanted wanna gonna got get gets getting know knows need needs
like try trying going make made say said use used see look think anyone anybody someone somebody something
anything everything nothing thing things way lot lots kind sort guy guys don doesn didn isn aren wasn weren hasn
haven hadn won wouldn shouldn couldn luckily fortunately unfortunately hopefully basically apparently huh
already each either simply without ask asked give tell feel bother tried wanting stuff ting
吗 嗎 什么 什麽 为什么 我应该 可以 的 了 是 告诉我
absolutely assuming awfully back bertanya both bothering certainly definitely despite doing everybody everyone
except folks hoping however implies kinda knowing means nope obviously own plenty probably putting quieres regarding
someday somehow sounds surely t thinking unless vreau whether wondered wondering
não почем تساعد يمكنك เหรอ ですか 不想 不是 不要 不过 为啥 也不想 了我的 什么呢
什么意思 他们 任何 任何人 但 但我 但现在 你 你应该 你的 关于 几次 即将 呢 咱们 哪个 啥 因为我
在我的 好吗 帮你 帮我 怎么 怎么做 怎么办 意味着 感觉自己 我不想 我们的 我可以 我没有 我的 我知道 我能
提问 教你 是不是 毕竟 没法 的回答 的答案 讓我 让你 让我 请您 这么多年 问我 问问
all wich
nhỉ даж お互 一封 不要太 为您 了吗 了吧 他 但如果 但我们 你怎么 去过 吧 当我 您 想知道 想要
我 我喜欢 我觉得 是否 是怎么 私は 谢谢 这句话 这笔
gotta
ってる 我希望 我要
""".split()
)

# Tokens no screen shows, whatever their rank. Readouts can surface profanity and sexual words, often from the
# adult-site spam among the web pages a tokenizer was built from, and other residue of those pages: broken
# encodings, invisible characters, page and code boilerplate. Some top the readout at a fifth of all positions
# whatever the message (experiments/noise_tokens.py), so they say nothing about it. No result depends on them:
# results use each case's own watch words.
UNSHOWN = frozenset(
    word.casefold()
    for word in """
motherfucker shit shits shitty bullshit crap crappy asshole bitch bitches slut sluts whore whores cunt dick dicks
cock cocks pussy xxx sex sexes sexe sexo seks sexy sext hentai fetish bdsm blowjob handjob nude nudes escort
escorts escorte geil geile horny boobs tits dildo incest onlyfans camgirl tjejer amatør salope salopes lesbische
lesbienne fag faggot danmark 婊 婊子 色情 淫 做爱
â ã ʺ ʼ ˈ ¹ ¾ ﬁ 聽 鈥 ㅤ ᆢ ㆍ 丨
继续访问 阅读全文 查看更多 例文帳に追加 advertisement forcanbeconvertedtoforeach tokenname propertyparams
outofboundsexception instanceof testdata
""".split()
)
UNSHOWN_PREFIXES = ("erot", "fuck", "milf", "porn")  # no innocent word starts like these


def strength(rank: int | None) -> str:
    """The band of a rank: "strong" (top 10), "weak" (top 100), "faint" (lower), or "none" when not read out."""
    if rank is None:
        return "none"
    if rank <= STRONG:
        return "strong"
    return "weak" if rank <= WEAK else "faint"


def alert(a: dict | None) -> dict | None:
    """A payload alert as the screens show it."""
    if a is None:
        return None
    return {
        "word": a["word"],
        "rank": a["rank"],
        "strength": strength(a["rank"]),
        "layer": a["layer"],
        "position": a["position"],
        "token": a["token"],
        "logit_rank": a["logit_rank"],
        "said_in_reply": a["said_in_reply"],
    }


def concepts_at(payload: dict, pos: int, glossary: Glossary) -> list[dict]:
    """The concepts read out at ``pos``, best first: one per word (variants such as "tool" / "tools" / "Tool"
    count once), leaving out function words, unshown tokens and words that only repeat the message.

    The payload's own ``echo`` flag also counts words of the reply. Here a word the model went on to say is kept:
    having it in mind before writing the reply is exactly what the screens show.
    """
    message = payload["user_text"].casefold()
    seen, out = set(), []
    for c in payload["concepts"][pos]:
        key = glossary.key(c["t"])
        if key in seen or echoes(c["t"], key, glossary, message) or is_stop(glossary, c["t"]) or is_unshown(c["t"]):
            continue
        seen.add(key)
        out.append(c)
    return out


def echoes(text: str, key: str, glossary: Glossary, message: str) -> bool:
    """An ASCII fragment of at most two characters, or a word the message contains (case-insensitive substring,
    as in ``engine.salient_concepts``), also through its English form ("口罩" -> "mask")."""
    word = text.strip().casefold()
    if len(word) <= 2 and word.isascii():
        return True
    return any(form and form in message for form in (word, glossary.english(text).casefold(), key))


def is_stop(glossary: Glossary, text: str) -> bool:
    word = glossary.english(text).casefold()
    return not word or word in STOPWORDS or glossary.key(text) in STOPWORDS


def is_unshown(text: str) -> bool:
    """Whether a token is one the screens never show (``UNSHOWN``, or a word that starts like ``UNSHOWN_PREFIXES``)."""
    word = text.strip().casefold()
    return word in UNSHOWN or word.startswith(UNSHOWN_PREFIXES)


def plain_key(glossary: Glossary, text: str) -> str:
    """A token's key as written: lower case and singular, without the dictionary's mapping of word pieces and other
    languages, so "Toxic" and "toxic" share a key but "可疑" is not "suspicious". Concern words are marked by it,
    because a result counts the English words themselves, not their equivalents."""
    word = text.strip().casefold()
    return word if word in glossary.entries else singular(word)


def concern_keys(glossary: Glossary, words: Iterable[str]) -> frozenset[str]:
    return frozenset(plain_key(glossary, w) for w in words)


def concept(c: dict, glossary: Glossary, concern: frozenset[str]) -> dict:
    """A payload concept as the screens show it, with its English and Japanese names."""
    return {
        "text": c["t"].strip(),
        "en": glossary.english(c["t"]),
        "ja": glossary.ja(c["t"]),
        "rank": c["r"],
        "layer": c["l"],
        "concern": plain_key(glossary, c["t"]) in concern,
    }


def strong_positions(payload: dict, words: Iterable[str]) -> dict[int, list[tuple[int, str, int]]]:
    """Where each of ``words`` ranked within the top 10 before the reply: {position: [(rank, word, layer), ...]},
    best first at each position (from the readout's per-position ranks)."""
    found: dict[int, list[tuple[int, str, int]]] = {}
    for word in words:
        for position, rank, layer in payload["watch"]["positions"].get(word, []):
            if rank <= STRONG:
                found.setdefault(position, []).append((rank, word, layer))
    return {position: sorted(hits) for position, hits in sorted(found.items())}
