from reedsolo import RSCodec, ReedSolomonError


def column_rs_encode(
    data_payloads: list[bytes],
    k: int,
    n: int,
) -> list[bytes]:
    rsc = RSCodec(n - k, nsize=n)
    max_len = max(len(p) for p in data_payloads)
    parity = [bytearray(max_len) for _ in range(n - k)]

    for col in range(max_len):
        column = bytearray(k)
        for i in range(k):
            column[i] = data_payloads[i][col] if col < len(data_payloads[i]) else 0
        encoded = rsc.encode(column)
        for j in range(n - k):
            parity[j][col] = encoded[k + j]

    return [bytes(p) for p in parity]


def column_rs_decode(
    data_payloads: list[bytes | None],
    parity_payloads: list[bytes | None],
    k: int,
    n: int,
) -> list[bytes] | None:
    if all(p is not None for p in data_payloads):
        return data_payloads

    missing = sum(1 for p in data_payloads if p is None)
    available_parity = sum(1 for p in parity_payloads if p is not None)
    if missing > available_parity:
        return None

    rsc = RSCodec(n - k, nsize=n)

    max_len = 0
    for p in list(data_payloads) + list(parity_payloads):
        if p and len(p) > max_len:
            max_len = len(p)

    recovered = [bytearray(max_len) for _ in range(k)]

    for col in range(max_len):
        received = bytearray(n)
        erasures: list[int] = []

        for i in range(k):
            if data_payloads[i] and col < len(data_payloads[i]):
                received[i] = data_payloads[i][col]
            else:
                erasures.append(i)

        for j in range(n - k):
            if parity_payloads[j] and col < len(parity_payloads[j]):
                received[k + j] = parity_payloads[j][col]
            else:
                erasures.append(k + j)

        try:
            decoded, _, _ = rsc.decode(received, erase_pos=erasures)
            for i in range(k):
                recovered[i][col] = decoded[i]
        except ReedSolomonError:
            return None

    return [bytes(p) for p in recovered]
