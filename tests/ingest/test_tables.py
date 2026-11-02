from ledgerlens.ingest.tables import (
    clean_cell,
    markdown_table_rows,
    normalize_rows,
    rows_to_markdown,
)


def test_edgar_split_cells_merge_into_clean_columns():
    rows = [
        ["", "", "", "2025", "", "", "", "2024", "", ""],
        ["Net sales", "", "", "$", "1,234", "", "", "$", "1,100", ""],
        ["Other income", "", "", "", "(321", ")", "", "", "45", ""],
        ["Gross margin", "", "", "", "40", "%", "", "", "38", "%"],
    ]
    md = rows_to_markdown(rows)
    assert md == (
        "|  | 2025 | 2024 |\n"
        "|---|---|---|\n"
        "| Net sales | $1,234 | $1,100 |\n"
        "| Other income | (321) | 45 |\n"
        "| Gross margin | 40% | 38% |"
    )


def test_inline_xbrl_spacing_is_tightened():
    assert clean_cell("( 321 )") == "(321)"
    assert clean_cell("$ 1,000") == "$1,000"
    assert clean_cell("a\xa0 b\n c") == "a b c"


def test_label_rows_above_data_stay_in_body():
    rows = [
        ["", "Years ended 2025", "2024"],
        ["Net sales:", "", ""],
        ["Products", "$10", "$9"],
    ]
    md = rows_to_markdown(rows)
    assert md is not None
    lines = md.splitlines()
    assert lines[0] == "|  | Years ended 2025 | 2024 |"
    assert lines[2] == "| Net sales: |  |  |"


def test_year_header_row_is_not_data():
    rows = [["Activity", "", ""], ["(in millions)", "2025", "2024"], ["Beginning", "$5", "$4"]]
    md = rows_to_markdown(rows)
    assert md is not None
    assert md.splitlines()[0] == "| Activity (in millions) | 2025 | 2024 |"


def test_layout_tables_are_not_markdown():
    assert rows_to_markdown([["Only one cell"]]) is None
    assert rows_to_markdown([["Item 7.", "MD&A"]]) is None  # single row
    assert rows_to_markdown([["•", "first point"], ["•", "second point"]]) is None


def test_empty_spacer_columns_dropped_and_ragged_rows_padded():
    rows = normalize_rows([["a", "", "1"], ["b", "", "2", ""], ["", "", ""]])
    assert rows == [["a", "1"], ["b", "2"]]


def test_pipe_escaping_round_trips():
    md = rows_to_markdown([["Name", "Value"], ["a|b", "1"]])
    assert md is not None
    assert markdown_table_rows(md) == [["Name", "Value"], ["a|b", "1"]]


def test_packed_value_columns_are_split():
    md = rows_to_markdown(
        [
            ["", "Employees", "Capital spending"],
            ["", "2018 2017", "2018 2017"],
            ["United States", "37,412 36,958", "$ 994 $ 852"],
            ["Asia Pacific", "18,971 18,283", "(5) —"],
        ]
    )
    assert md is not None
    lines = md.splitlines()
    assert lines[2] == "| United States | 37,412 | 36,958 | $994 | $852 |"
    assert lines[3] == "| Asia Pacific | 18,971 | 18,283 | (5) | — |"


def test_ranges_and_irregular_columns_are_not_split():
    from ledgerlens.ingest.tables import packed_cell, split_packed_columns

    assert not packed_cell("2025 – 2062")
    assert packed_cell("1,624 1,672")
    rows = [["Notes", "2025 – 2062"], ["Bonds", "2028 – 2035"]]
    assert split_packed_columns([r[:] for r in rows]) == rows
    mixed = [["a", "1 2"], ["b", "3"]]  # one cell has a single value: ambiguous
    assert split_packed_columns([r[:] for r in mixed]) == mixed
