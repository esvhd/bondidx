import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Union, List, Iterable
import tqdm
import seaborn as sns
import xarray as xr

import matrix.plottools as pts
import matrix.config as mc
import matrix.risk as risk
from matrix.timeseries import rebase
from matrix.utils import weighted_average

import pypbo.perf as perf


@dataclass(kw_only=True)
class IndexConfig(object):
    oasd: str
    oas: str
    ytw: str
    price: str
    maturity: str
    coupon: str
    ticker: str
    rating: str
    rating_bucket: str = "rating_bucket"
    isin: str = None
    ticker: str = None
    seniority: str = None
    amtout: str = None
    country: str = None
    currency: str = None
    sector_lvl_1: str = None
    sector_lvl_2: str = None
    sector_lvl_3: str = None
    sector_lvl_4: str = None
    weight: str = None


def load_config(config_path: str, config_name: str) -> IndexConfig:
    """Example:
    conf = idx.load_config('~/config/baml_mapping.ini', config_name='DEFAULT')

    Parameters
    ----------
    config_path : str
        _description_
    config_name : str
        _description_

    Returns
    -------
    IndexConfig
        _description_
    """
    # load config file
    conf = mc.load(config_path)
    idx_conf = conf[config_name]

    conf = IndexConfig(**idx_conf)
    return conf


def slice_reweight(df: pd.DataFrame) -> pd.DataFrame:
    """Reweight bond slice grouped by date

    Parameters
    ----------
    df : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    new_wgts = df.groupby("date").pct_weight.transform(lambda x: x / x.sum())
    out = df.assign(new_weight=new_wgts)
    return out


def _slice_field_in(df: pd.DataFrame, field: str, values: Union[Iterable, str]):
    if values is None:
        return df
    if isinstance(values, str):
        values = [values]
    mask = df[field].isin(values)
    return df.loc[mask]


def _slice_field_not_in(
    df: pd.DataFrame, field: str, values: Union[Iterable, str]
):
    if values is None:
        return df
    if isinstance(values, str):
        values = [values]
    mask = df[field].isin(values)
    return df.loc[~mask]


def slice_baml(
    df: pd.DataFrame,
    config: IndexConfig,
    sector_level_2: Union[str, Iterable[str]] = None,
    sector_level_3: Union[str, Iterable[str]] = None,
    rating_bucket: Union[str, Iterable[str]] = None,
    min_maturity: float = None,
    max_maturity: float = None,
    min_eff_dur: float = None,
    max_eff_dur: float = None,
    rating_notched: Union[str, Iterable[str]] = None,
    min_oas: float = None,
    remove_defaulted: bool = False,
    seniority: Union[str, Iterable[str]] = None,
    seniority_not_in: Union[str, Iterable[str]] = None,
    country_in: Union[str, Iterable[str]] = None,
    country_not_in: Union[str, Iterable[str]] = None,
    reweight: bool = True,
) -> pd.DataFrame:
    """Slice BAML bond data frame by various criteria.

    Min / Max type of conditions are always in the form of (min, max].

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    rating_bucket : str, optional
        _description_, by default None
    min_maturity : float, optional
        _description_, by default None
    max_maturity : float, optional
        _description_, by default None
    min_eff_dur : float, optional
        _description_, by default None
    max_eff_dur : float, optional
        _description_, by default None
    rating_notched : str, optional
        _description_, by default None

    Returns
    -------
    pd.DataFrame
        _description_
    """
    out = df

    # if sector_level_2:
    #     if isinstance(sector_level_2, str):
    #         sector_level_2 = [sector_level_2]
    #     out = out.loc[out.sector_level_2.isin(sector_level_2)]

    # if sector_level_3:
    #     if isinstance(sector_level_3, str):
    #         sector_level_3 = [sector_level_3]
    #     out = out.loc[out.sector_level_3.isin(sector_level_3)]

    # if rating_bucket:
    #     out = out.query(f"rating_bucket == '{rating_bucket}'")

    # if rating_notched:
    #     out = out.query(f"composite_rating == '{rating_notched}'")

    out = _slice_field_in(out, config.sector_lvl_2, sector_level_2)
    out = _slice_field_in(out, config.sector_lvl_3, sector_level_3)

    out = _slice_field_in(out, config.rating_bucket, rating_bucket)
    out = _slice_field_in(out, config.rating, rating_notched)

    if min_maturity:
        out = out.query(f"years_to_mat > {min_maturity}")

    if max_maturity:
        out = out.query(f"years_to_mat <= {max_maturity}")

    if min_eff_dur:
        out = out.query(f"{config.oasd} > {min_eff_dur}")

    if max_eff_dur:
        out = out.query(f"{config.oasd} <= {max_eff_dur}")

    if min_oas is not None:
        out = out.query(f"{config.oas} > {min_oas}")

    # if seniority is not None:
    #     if isinstance(seniority, Iterable):
    #         out = out.loc[out["type"].isin(seniority)].copy()
    #     else:
    #         out = out.query(f"type == '{seniority}'")

    out = _slice_field_in(out, config.seniority, seniority)
    out = _slice_field_not_in(out, config.seniority, seniority_not_in)

    out = _slice_field_in(out, config.country, country_in)
    out = _slice_field_not_in(out, config.country, country_not_in)

    if remove_defaulted:
        # proxy for identifying defaulted bond is OAS vs Govt == 10,000
        out = out.query(f"{config.oas} < 10000")

    if reweight:
        out = slice_reweight(out)

    return out


def slice_to_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Generate time series data for a slice of BAML bonds in long format.

    Parameters
    ----------
    df : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # pct = df.pivot(index="date", columns="isin", values="excess_rtn_pct_mtd")
    # oas = df.pivot(index="date", columns="isin", values="oas_vs_govt")
    # dur = df.pivot(index="date", columns="isin", values="effective_duration")
    # wgt = df.pivot(index="date", columns="isin", values="new_weight")

    pct = pd.pivot_table(
        df, index="date", columns="isin", values="excess_rtn_pct_mtd"
    )
    oas = pd.pivot_table(df, index="date", columns="isin", values="oas_vs_govt")
    ytw = pd.pivot_table(
        df, index="date", columns="isin", values="yield_to_worst"
    )
    dur = pd.pivot_table(
        df, index="date", columns="isin", values="effective_duration"
    )
    wgt = pd.pivot_table(df, index="date", columns="isin", values="new_weight")

    # pct = pct.reindex_like(wgt)
    # oas = pct.reindex_like(wgt)
    # dur = pct.reindex_like(wgt)

    n_tickers = df.groupby("date").ticker.nunique()
    n_isins = df.groupby("date")["isin"].nunique()

    data = dict()
    data["excess_pct"] = (wgt * pct).sum(axis=1) / 100.0
    data["oas"] = (wgt * oas).sum(axis=1)
    data["ytw"] = (wgt * ytw).sum(axis=1)
    data["eff_dur"] = (wgt * dur).sum(axis=1)
    data["excess_log"] = perf.pct_to_log_return(data["excess_pct"])
    data["dts"] = (wgt * dur * oas).sum(axis=1)
    data["ticker_count"] = n_tickers
    data["bond_count"] = n_isins

    return pd.DataFrame.from_dict(data)


@dataclass(kw_only=True)
class BondIndexSlice(object):
    config: IndexConfig
    sector_level_2: Union[str, Iterable[str]] = None
    sector_level_3: Union[str, Iterable[str]] = None
    rating_bucket: str = None
    rating_notched: str = None
    min_maturity: float = None
    max_maturity: float = None
    min_eff_dur: float = None
    max_eff_dur: float = None
    seniority: Union[str, Iterable[str]] = None
    seniority_not_in: Union[str, Iterable[str]] = None
    country_in: Union[str, Iterable[str]] = None
    country_not_in: Union[str, Iterable[str]] = None
    reweight: bool = True
    name: str = None

    def slice(
        self,
        df: pd.DataFrame,
        remove_defaulted: bool = False,
        min_oas: float = None,
    ) -> pd.DataFrame:
        out = slice_baml(
            df,
            self.config,
            sector_level_2=self.sector_level_2,
            sector_level_3=self.sector_level_3,
            rating_bucket=self.rating_bucket,
            rating_notched=self.rating_notched,
            min_maturity=self.min_maturity,
            max_maturity=self.max_maturity,
            min_eff_dur=self.min_eff_dur,
            max_eff_dur=self.max_eff_dur,
            reweight=self.reweight,
            seniority=self.seniority,
            seniority_not_in=self.seniority_not_in,
            country_in=self.country_in,
            country_not_in=self.country_not_in,
            min_oas=min_oas,
            remove_defaulted=remove_defaulted,
        )
        return out

    def get_name(self) -> str:
        if self.name is not None:
            return self.name

        out = ""

        if self.sector_level_2 is not None:
            if isinstance(self.sector_level_2, str):
                out += self.sector_level_2 + "_"
            else:
                sector_tags = "_".join(self.sector_level_2)
                out += sector_tags + "_"
        if self.sector_level_3 is not None:
            if isinstance(self.sector_level_3, str):
                out += self.sector_level_3 + "_"
            else:
                sector_tags = "_".join(self.sector_level_3)
                out += sector_tags + "_"

        if self.rating_notched is not None or self.rating_bucket is not None:
            out += (
                self.rating_bucket
                if self.rating_bucket
                else self.rating_notched
            ) + "_"

        min_field = self.min_eff_dur if self.min_eff_dur else self.min_maturity
        max_field = self.max_eff_dur if self.max_eff_dur else self.max_maturity

        if min_field is not None:
            out += (
                f"0{min_field:.0f}_" if min_field < 10 else f"{min_field:.0f}_"
            )

        if max_field is not None:
            out += f"0{max_field:.0f}" if max_field < 10 else f"{max_field:.0f}"

        self.name = out
        return out

    def slice_groupby_mean(
        self,
        data: pd.DataFrame,
        weight_col: str = "pct_weight",
        groupby=["date", "ticker"],
        value_col="oas_vs_govt",
        **slice_kws,
    ) -> pd.Series:
        """Slice, groupby date and ticker, take weighted average oas based on
        bond weights, then take average oas across all tickers with equal weight.

        Parameters
        ----------
        data : pd.DataFrame
            _description_
        weight_col : str, optional
            _description_, by default "pct_weight"
        groupby : list, optional
            _description_, by default ["date", "ticker"]
        value_col : str, optional
            _description_, by default "oas_vs_govt"

        Returns
        -------
        pd.Series
            _description_
        """
        df = self.slice(df=data, **slice_kws)
        # return df
        value = weighted_average(
            df, data_col=value_col, weight_col=weight_col, by_col=groupby
        ).reset_index()
        value.columns = groupby + [value_col]
        out = value.groupby("date").mean(numeric_only=True)
        return out


@dataclass
class BondIndexStudy(object):
    annual_factor: int
    time_series_data: dict = None
    members: dict = None

    @staticmethod
    def get_default_slices(config: IndexConfig) -> List[BondIndexSlice]:
        # fmt: off
        slices = []

        slices.append(BondIndexSlice(config=config, name='_market', reweight=True))

        # maturity slices
        slices.append(BondIndexSlice(config=config, min_maturity=1., max_maturity=3., reweight=True))
        slices.append(BondIndexSlice(config=config, min_maturity=1., max_maturity=5., reweight=True))
        slices.append(BondIndexSlice(config=config, min_maturity=1., max_maturity=7., reweight=True))
        slices.append(BondIndexSlice(config=config, min_maturity=1., max_maturity=10., reweight=True))
        slices.append(BondIndexSlice(config=config, min_maturity=3., max_maturity=5., reweight=True))
        slices.append(BondIndexSlice(config=config, min_maturity=5., max_maturity=10., reweight=True))

        # rating slices
        slices.append(BondIndexSlice(config=config, rating_bucket='A', reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='A', min_maturity=1., max_maturity=5., reweight=True))
        # slices.append(BondIndexSlice(config=config, rating_bucket='A', min_maturity=3., max_maturity=5., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='A', min_maturity=5., max_maturity=10., reweight=True))
        # slices.append(BondIndexSlice(config=config, rating_bucket='A', min_maturity=7., max_maturity=10., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='A', min_maturity=10., max_maturity=15., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='A', min_maturity=15., max_maturity=20., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='A', min_maturity=20., max_maturity=31., reweight=True))

        slices.append(BondIndexSlice(config=config, rating_bucket='BBB', reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='BBB', min_maturity=1., max_maturity=5., reweight=True))
        # slices.append(BondIndexSlice(config=config, rating_bucket='BBB', min_maturity=3., max_maturity=5., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='BBB', min_maturity=5., max_maturity=10., reweight=True))
        # slices.append(BondIndexSlice(config=config, rating_bucket='BBB', min_maturity=7., max_maturity=10., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='BBB', min_maturity=10., max_maturity=15., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='BBB', min_maturity=15., max_maturity=20., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='BBB', min_maturity=20., max_maturity=31., reweight=True))

        slices.append(BondIndexSlice(config=config, rating_bucket='BB', reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='BB', min_maturity=1., max_maturity=5., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='BB', min_maturity=5., max_maturity=10., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='B', reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='B', min_maturity=1., max_maturity=5., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='B', min_maturity=5., max_maturity=10., reweight=True))
        slices.append(BondIndexSlice(config=config, rating_bucket='CCC', min_maturity=1., max_maturity=5., reweight=True))
        # fmt: on

        sectors = [
            "Basic Industry",
            "Banking",
            "Transportation",
            "Telecommunications",
            "Capital Goods",
            "Financial Services",
            "Energy",
            "Automotive",
            "Utility",
            "Consumer Goods",
            "Services",
            "Insurance",
            "Healthcare",
            "Real Estate",
            "Media",
            "Technology & Electronics",
            "Retail",
            "Leisure",
        ]
        for ss in sectors:
            slices.append(
                BondIndexSlice(config=config, sector_level_3=ss, reweight=True)
            )

        # lvl2 = ['Industrials', 'Financial', 'Utility']
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Industrials",
                rating_bucket="BBB",
                reweight=True,
            )
        )
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Financial",
                rating_bucket="BBB",
                reweight=True,
            )
        )
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Utility",
                rating_bucket="BBB",
                reweight=True,
            )
        )

        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Industrials",
                rating_bucket="A",
                reweight=True,
            )
        )
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Financial",
                rating_bucket="A",
                reweight=True,
            )
        )
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Utility",
                rating_bucket="A",
                reweight=True,
            )
        )

        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Industrials",
                rating_bucket="BB",
                reweight=True,
            )
        )
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Financial",
                rating_bucket="BB",
                reweight=True,
            )
        )
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Utility",
                rating_bucket="BB",
                reweight=True,
            )
        )

        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Industrials",
                rating_bucket="B",
                reweight=True,
            )
        )
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Financial",
                rating_bucket="B",
                reweight=True,
            )
        )
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2="Utility",
                rating_bucket="B",
                reweight=True,
            )
        )

        # seniority = [
        #     ['SENR', 'SECR', 'SNPR'],
        #     ['T2', 'UT2'],
        #     ['SUB', 'JSUB', 'T1', 'PFD', 'AT1'],
        # ]

        # Bank seniors
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_3="Banking",
                name="Bank_Seniors",
                seniority=["SENR", "SECR", "SNPR"],
                reweight=True,
            )
        )
        # Bank T2
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_3="Banking",
                name="Bank_T2",
                seniority=["T2", "UT2"],
                reweight=True,
            )
        )
        # Bank T1, in US C0A0 index, this universe is very small since most US banks now issue preferrds
        # which are included in a separate index
        # TODO: may need to remove SUB from tier 1 definition
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_3="Banking",
                name="Bank_T1",
                seniority=["SUB", "JSUB", "T1", "PFD", "AT1"],
                reweight=True,
            )
        )
        # Insurance RT1
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_3="Insurance",
                name="Insurance_T1",
                seniority=["SUB", "JSUB", "T1", "PFD", "AT1"],
                reweight=True,
            )
        )
        # Insurance T2
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_3="Insurance",
                name="Insurance_T2",
                seniority=["T2", "UT2"],
                reweight=True,
            )
        )
        # corporate hybrids
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2=["Utility", "Industrials"],
                name="Corp_Hybrids",
                seniority=["SUB", "JSUB", "PFD"],
                reweight=True,
            )
        )
        # corporate seniors
        slices.append(
            BondIndexSlice(
                config=config,
                sector_level_2=["Utility", "Industrials"],
                name="Corp_Seniors",
                seniority=["SENR", "SECR", "SNPR"],
                reweight=True,
            )
        )

        return slices

    def columns(self):
        if self.time_series_data is None:
            return None

        for k, v in self.time_series_data.items():
            if v is not None:
                return v.columns

        return None

    def analyse(
        self,
        df: pd.DataFrame,
        slices: Iterable[BondIndexSlice] = None,
        config: IndexConfig = None,
        min_size: int = 10,
        verbose: bool = False,
        **slice_kws,
    ):
        """Generate slice data. For studying general market OAS / yield history,
        one should add the following slice_kws:
            remove_defaulted=True, min_oas=15
        This removes noice from defaulted bonds with 10k OAS, as well as some
        bonds trading through UST, which could be either corporate action
        or bad data.

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        slices : Iterable[BondIndexSlice], optional
            _description_, by default None
        config: IndexConfig, optional
            must be given if slices is None
        """
        if slices is None:
            slices = BondIndexStudy.get_default_slices(config)

        # for each slice, get time series stats
        res = dict()
        memb = dict()
        for x in tqdm.tqdm(slices):
            out = x.slice(df, **slice_kws)
            stats = slice_to_ts(out)
            name = x.get_name()
            if verbose:
                print(f"{name} -> rows # = {len(out)}, ts row # = {len(stats)}")
            if len(out) < min_size:
                continue
            res[name] = stats
            memb[name] = out

        # for some reason the below verison is very slow...
        # funcs = [x.slice for x in slices]
        # names = [x.get_name() for x in slices]
        # results = hp.parallel_func(funcs, df, **slice_kws)
        # for out, name in zip(results, names):
        #     stats = slice_to_ts(out)
        #     res[name] = stats
        #     memb[name] = out

        self.members = memb
        self.time_series_data = res

    def show_names(self):
        if self.members is None:
            raise ValueError("Must call analyse() first")

        return self.members.keys()

    def get_slice(self, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.members is None:
            raise ValueError("Must call analyse() first")

        return self.time_series_data.get(name), self.members.get(name)

    def get_column(self, col: str) -> pd.DataFrame:
        """Get a feature column from aggregated stats data.
        Aggregate stats: stats for each slice with bonds reweighted.

        Parameters
        ----------
        col : str
            _description_

        Returns
        -------
        pd.DataFrame
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if self.time_series_data is None:
            raise ValueError("Must call analyse() first")

        rtns = dict()
        for key, val in self.time_series_data.items():
            rtns[key] = val[col]
        rtns = pd.DataFrame.from_dict(rtns)
        return rtns

    def get_returns(self) -> pd.DataFrame:
        return self.get_column(col="excess_log")

    def get_risk_stats(self, tail: int = None) -> pd.DataFrame:
        rtns = self.get_returns()
        if tail is not None:
            rtns = rtns.tail(tail)
        stats = risk.get_risk_stats(rtns, factor=self.annual_factor)
        return stats

    def plot(
        self, x="sharpe_non_iid", y="calmar", tail: int = None, figsize=(7, 6)
    ):
        stats = self.get_risk_stats(tail=tail)
        # sort so that labels are added in the correct alternating position
        stats = stats.sort_values(x)

        _, ax = pts.subplots(figsize=figsize)
        sns.scatterplot(x=x, y=y, data=stats, ax=ax)
        pts.scatter_label(
            labels=stats.index,
            x_values=stats[x],
            y_values=stats[y],
            ax=ax,
            alter_loc=True,
        )
        return ax

    def plot_nav(self, tail: int = None) -> pd.DataFrame:
        rtns = self.get_returns()
        if tail is not None:
            rtns = rtns.tail(tail)
        nav = rebase(np.exp(rtns.cumsum()))
        nav = pts.plot_sorted(nav)
        return nav

    def plot_oas(self, tail: int = None) -> pd.DataFrame:
        ds = xr.Dataset(self.time_series_data)
        oas = ds.sel(dim_1="oas").to_dataframe().drop(columns="dim_1")
        if tail is not None:
            oas = oas.tail(tail)
        oas = pts.plot_sorted(oas)
        return oas

    def change_summary(self):
        cols = ["oas", "ytw", "eff_dur"]
        data = dict()
        ds = xr.Dataset(self.time_series_data)
        for cc in cols:
            df = ds.sel(dim_1=cc).to_dataframe().drop(columns="dim_1")
            chg1 = df.diff().iloc[[-1]].T
            chg3 = df.diff(3).iloc[[-1]].T
            chg6 = df.diff(6).iloc[[-1]].T
            view = df.iloc[[-1]].T

            view.columns = [cc]
            chg1.columns = [cc + "_chg1"]
            chg3.columns = [cc + "_chg3"]
            chg6.columns = [cc + "_chg6"]
            view = (
                view.join(chg1, how="left")
                .join(chg3, how="left")
                .join(chg6, how="left")
            )
            data[cc] = view

        dts = ds.sel(dim_1="dts").to_dataframe().drop(columns="dim_1")
        dts = dts.iloc[[-1]].T.round(0)
        dts.columns = ["dts"]
        if "_market" in dts.index:
            dts = dts.assign(
                dts_ratio=(dts.dts / dts.loc["_market", "dts"]).round(2)
            )

        out = data.get("oas")
        if out is not None:
            out = dts.join(out.round(0), how="right")
            ytw = data.get("ytw")
            if ytw is not None:
                out = out.join(ytw.round(2), how="left")
        if out is not None:
            ytw = data.get("eff_dur")
            if ytw is not None:
                out = out.join(ytw.round(2), how="left")
        return out.sort_index()
