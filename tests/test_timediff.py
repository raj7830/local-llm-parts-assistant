from datetime import date

import pytest

from src.timediff import business_hours_between, hours_between


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("2022/02/15 00:05", "2022/02/15 01:00", 1),  # example from the task
        ("2017/05/13 12:00", "2017/05/13 12:00", 0),
        ("2017/05/13 12:00", "2017/05/13 12:29", 0),
        ("2017/05/13 12:00", "2017/05/13 12:30", 1),  # half rounds up
        ("2017/05/13 12:00", "2017/05/14 12:00", 24),
        ("2022/02/15 01:00", "2022/02/15 00:05", 1),  # argument order irrelevant
        ("2020/02/28 23:00", "2020/03/01 01:00", 26),  # leap day
    ],
)
def test_hours_between(start, end, expected):
    assert hours_between(start, end) == expected


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("2022/02/15 10:00", "2022/02/15 12:00", 2),  # inside working hours
        ("2022/02/15 07:00", "2022/02/15 20:00", 8),  # clipped to 09-17
        ("2022/02/15 18:00", "2022/02/16 08:00", 0),  # overnight
        ("2022/02/15 16:00", "2022/02/16 10:00", 2),  # Tue 16-17 + Wed 9-10
        ("2022/02/18 16:00", "2022/02/21 10:00", 2),  # Fri -> Mon skips weekend
        ("2022/02/19 10:00", "2022/02/20 15:00", 0),  # Sat -> Sun
        ("2022/02/14 09:00", "2022/02/20 23:59", 40),  # one full week
        ("2022/02/15 09:00", "2022/02/15 09:29", 0),
        ("2022/02/15 09:00", "2022/02/15 09:30", 1),
        ("2022/02/16 10:00", "2022/02/15 16:00", 2),  # reversed order
    ],
)
def test_business_hours_between(start, end, expected):
    assert business_hours_between(start, end) == expected


def test_business_hours_with_holidays():
    assert business_hours_between(
        "2022/12/23 09:00", "2022/12/27 17:00", holidays=[date(2022, 12, 26)]
    ) == 16


def test_invalid_input():
    with pytest.raises(ValueError):
        hours_between("2022-02-15 00:05", "2022/02/15 01:00")
