#! /usr/bin/python

import argparse
from datetime import datetime, timedelta
from os.path import join, isfile
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def date_from_cmd(date_str, at_noon=True):
    hours = 12 if at_noon else 0
    try:
        return datetime.strptime(date_str, "%Y%m%d").replace(
            hour=hours, minute=0, second=0
        )
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{
                date_str}'. Expected YYYYMMDD."
        )


class myFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    pass


parser = argparse.ArgumentParser(formatter_class=myFormatter)

parser.add_argument(
    "-s",
    "--station",
    metavar="STATION",
    dest="station",
    default=[],
    required=False,
    nargs="+",
    help="Name of station(s), as in the respective .mom files to be read.",
)

parser.add_argument(
    "-d",
    "--data-dir",
    metavar="DATA_DIR",
    dest="data_dir",
    default="raw_files",
    required=False,
    help="Directory where the *.mom files are located.",
)

parser.add_argument(
    "-f",
    "--from",
    metavar="FROM_EPOCH",
    dest="tmin",
    default=datetime.min,
    required=False,
    type=date_from_cmd,
    help="Starting date for plotting as YYYYMMDD. Default is first epoch found in mom files.",
)

parser.add_argument(
    "-t",
    "--to",
    metavar="TO_EPOCH",
    dest="tmax",
    default=datetime.max,
    required=False,
    type=date_from_cmd,
    help="Ending date for displacement as YYYYMMDD. Default is last epoch found in mom files.",
)

parser.add_argument(
    "--mstyle",
    metavar="STYLE",
    dest="style_sheet",
    default=None,
    required=False,
    help="Set the style sheet of the generated plot. To see a full list of available styles, use: import matplotlib.pyplot as plt; print(plt.style.available)",
)

parser.add_argument(
    "-l",
    "--load-config",
    metavar="CONFIG",
    dest="config",
    default=None,
    required=False,
    help="Load a (plot) configuration file.",
)

parser.add_argument(
    "--no-breaks",
    action="store_true",
    dest="no_breaks",
    help="Do not plot breaks and/or offsets (by default they are plotted as vertical lines).",
)

parser.add_argument(
    "--add-model",
    action="store_true",
    dest="plot_model",
    help="Add model line to the plot(s). In this case, the corresponding .out files.",
)

parser.add_argument(
    "-c",
    "--default-config",
    action="store_true",
    dest="spit_config",
    help="Print the (default) configuration file and exit.",
)


def mjd2t(mjd):
    # MJD zero point: 1858-11-17 00:00:00 UTC
    mjd_start = datetime(1858, 11, 17)
    # Convert MJD to datetime
    return mjd_start + timedelta(days=mjd)


def feedModel(fn, breaks, offsets=[]):
    trends = []
    harmonics = []
    jumps = []
    jfrom = datetime.min
    jidx = 0
    breaks.append(datetime.max)  # end time of last interval

    def harmonicsAdd(hlist, freq, key, value):
        for h in hlist:
            if h["freq"] == freq:
                h[key] = value
                return hlist
        hlist.append({"freq": freq, key: value})
        return hlist

    with open(fn, "r") as fin:
        for line in fin.readlines():
            if line.startswith("bias :"):
                bias = float(line.split()[2])
            elif line.startswith("trend:"):
                trends.append(
                    {"from": jfrom, "to": breaks[jidx], "value": float(line.split()[1])}
                )
                jfrom = breaks[jidx]
                jidx += 1
            elif line.startswith("cos"):
                freq = float(line.split()[1])
                camp = float(line.split()[3])
                harmonics = harmonicsAdd(harmonics, 365.25 / freq, "camp", camp)
            elif line.startswith("sin"):
                freq = float(line.split()[1])
                samp = float(line.split()[3])
                harmonics = harmonicsAdd(harmonics, 365.25 / freq, "samp", samp)
    return datetime.min, bias, trends, harmonics, jumps


def feedComponent(fn):
    def parse_data_line(line):
        vals = [float(x) for x in line.split()]
        return (
            (vals[0], vals[1], vals[2]) if len(vals) == 3 else (vals[0], vals[1], None)
        )

    t = []
    x = []
    mx = []
    breaks = []
    offsets = []
    with open(fn, "r") as fin:
        for line in fin.readlines():
            if line[0] != "#":
                tmjd, val, mval = parse_data_line(line)
                t.append(mjd2t(tmjd))
                x.append(val)
                mx.append(mval)
            else:
                if line.startswith("# break"):
                    breaks.append(mjd2t(float(line.split()[2])))
                elif line.startswith("# offset"):
                    offsets.append(mjd2t(float(line.split()[2])))
    mx = [] if all(y is None for y in mx) else mx
    return t, x, mx, breaks, offsets


class HectorModel:
    def __init__(self, t0, bias, trends, harmonics, jumps):
        self.t0, self.bias, self.trends, self.harmonics, self.jumps = (
            t0,
            bias,
            trends,
            harmonics,
            jumps,
        )

    def setStartEpoch(self, t0):
        for trend in self.trends:
            if trend["from"] == datetime.min:
                trend["from"] = t0

    @classmethod
    def fromMom(self, fn: str, breaks: list):
        return HectorModel(*feedModel(fn, breaks))

    def at(self, t):
        val = self.bias
        for trend in self.trends:
            if t >= trend["from"] and t < trend["to"]:
                val += (t - trend["from"]).days / 365.25e0 * trend["value"]
            elif t >= trend["from"]:
                val += (trend["to"] - trend["from"]).days / 365.25e0 * trend["value"]
        for hrmn in self.harmonics:
            omega = 2e0 * np.pi * hrmn["freq"]
            val += hrmn["camp"] * np.cos(omega * (t - self.t0).days / 365.25e0) + hrmn[
                "samp"
            ] * np.sin(omega * (t - self.t0).days / 365.25e0)
        return val

    def values(self, tarray):
        return [self.at(t) for t in tarray]


class HectorTsModel:
    def __init__(self, site, northc, eastc, upc):
        self.site = site
        self.north, self.east, self.up = northc, eastc, upc

    def setStartEpoch(self, t0):
        self.north.setStartEpoch(t0)
        self.east.setStartEpoch(t0)
        self.up.setStartEpoch(t0)

    @classmethod
    def fromMom(self, site, data_dir, bn, be, bu):
        return HectorTsModel(
            site,
            HectorModel.fromMom(join(data_dir, site + "_0.out"), bn),
            HectorModel.fromMom(join(data_dir, site + "_1.out"), be),
            HectorModel.fromMom(join(data_dir, site + "_2.out"), bu),
        )

    def values(self, comp: str, tmin, tmax, every_hours=24):
        tarray = []
        t = tmin
        while t < tmax:
            tarray.append(t)
            t += timedelta(hours=every_hours)
        if comp == "north":
            return tarray, self.north.values(tarray)
        if comp == "east":
            return tarray, self.east.values(tarray)
        if comp == "up":
            return tarray, self.up.values(tarray)


class HectorTsComponent:
    def __init__(self, t: list, x: list, mx: list, breaks: list, offsets: list):
        self.t, self.x, self.mx, self.breaks, self.offsets = t, x, mx, breaks, offsets

    def lowerBound(self, t):
        if self.t[0] >= t:
            return 0
        if t >= self.t[-1]:
            return len(self.t) - 1
        for j, tj in enumerate(self.t):
            if tj >= t:
                return j
        raise RuntimeError("Something went wrong in HectorTsComponent::lowerBound()")

    @classmethod
    def fromMom(self, fn: str):
        return HectorTsComponent(*feedComponent(fn))

    def assertSorted(self):
        for j, t in enumerate(self.t[1:]):
            assert t > self.t[j]

    def assertRange(self, tmin, tmax):
        for t in self.t:
            assert t >= tmin and t < tmax

    def filterBreaks(self, tmin, tmax):
        return [b for b in self.breaks if (b >= tmin and b < tmax)]

    def filterOffsets(self, tmin, tmax):
        return [b for b in self.offsets if (b >= tmin and b < tmax)]

    def filter(self, tmin=datetime.min, tmax=datetime.max):
        self.assertSorted()
        t0_idx = self.lowerBound(tmin) if tmin is not datetime.min else 0
        t1_idx = (
            self.lowerBound(tmax) + 1 if tmax is not datetime.max else len(self.t) - 1
        )
        return HectorTsComponent(
            self.t[t0_idx:t1_idx],
            self.x[t0_idx:t1_idx],
            self.mx[t0_idx:t1_idx] if (self.mx != []) else [],
            self.filterBreaks(tmin, tmax),
            self.filterOffsets(tmin, tmax),
        )

    def tLimits(self):
        return self.t[0], self.t[-1]


class HectorTs:
    def __init__(self, site, northc, eastc, upc):
        self.site = site
        self.north, self.east, self.up = northc, eastc, upc

    @classmethod
    def fromMom(self, site, data_dir):
        return HectorTs(
            site,
            HectorTsComponent.fromMom(join(data_dir, site + "_0.mom")),
            HectorTsComponent.fromMom(join(data_dir, site + "_1.mom")),
            HectorTsComponent.fromMom(join(data_dir, site + "_2.mom")),
        )

    def breaks(self):
        return self.north.breaks, self.east.breaks, self.up.breaks

    def filter(self, tmin=datetime.min, tmax=datetime.max):
        return HectorTs(
            self.site,
            self.north.filter(tmin, tmax),
            self.east.filter(tmin, tmax),
            self.up.filter(tmin, tmax),
        )

    def assertRange(self, tmin, tmax):
        assert (
            self.north.assertRange(tmin, tmax)
            and self.east.assertRange(tmin, tmax)
            and self.up.assertRange(tmin, tmax)
        )

    def tLimits(self):
        tmin = min(
            self.north.tLimits()[0], self.east.tLimits()[0], self.up.tLimits()[0]
        )
        tmax = max(
            self.north.tLimits()[1], self.east.tLimits()[1], self.up.tLimits()[1]
        )
        return tmin, tmax

    def startEpoch(self):
        return self.tLimits()[0]


# PlotOptions:
plotOptions = {
    "markerSize": {
        "value": 1,
        "type": "po",
        "var": float,
        "help": "Marker size for scatter plot(s).",
    },
    "everyYears": {
        "value": 1,
        "type": "po",
        "var": int,
        "help": "Major ticks will be plotted every 'everyYears' years.",
    },
    "everyMonths": {
        "value": 2,
        "type": "po",
        "var": int,
        "help": "Minor ticks will be plotted every 'everyMonths' months.",
    },
    "xTitleSize": {
        "value": 14,
        "type": "po",
        "var": int,
        "help": "Label font size for x-axis.",
    },
    "yTitleSize": {
        "value": 12,
        "type": "po",
        "var": int,
        "help": "Label font size for y-axis.",
    },
    "titleSize": {"value": 18, "type": "po", "var": int, "help": "Title font size."},
    "legendPosition": {
        "value": "lower left",
        "type": "po",
        "var": str,
        "help": "Position of legend in (u-component) sub-plot.",
    },
    "legendFontSize": {
        "value": 10,
        "type": "po",
        "var": int,
        "help": "Font size for legend.",
    },
    "legendMarkerSize": {
        "value": 10,
        "type": "po",
        "var": float,
        "help": "Marker size size for legend.",
    },
    "modelLineWidth": {
        "value": 0.6,
        "type": "po",
        "var": float,
        "help": "Line width for plotting model line(s).",
    },
    "font.family": {
        "value": "monospace",
        "type": "rc",
        "var": str,
        "help": "Font family for text and labels.",
    },
    "font.monospace": {
        "value": "DejaVu Sans Mono",
        "type": "rc",
        "var": str,
        "help": "Font name for the specific font family set (see 'font.family')",
    },
}


def loadConfig(fn):
    with open(fn, "r") as fin:
        for line in fin.readlines():
            if not line.startswith("#"):
                l = line.split()
                key = l[0]
                value = " ".join(l[1:])
                dct = plotOptions[key]
                dct["value"] = dct["var"](value.strip())
                if dct["type"] == "rc":
                    plt.rcParams[key] = dct["value"]


def spitConfig():
    for k, v in plotOptions.items():
        print("# {:}".format(v["help"]))
        print("{:} {}".format(k, v["value"]))


if __name__ == "__main__":

    # parse cmd
    args = parser.parse_args()

    # do we only need to spit a configuration file ?
    if args.spit_config:
        spitConfig()
        sys.exit(0)
    else:
        if len(args.station) == 0:
            print("No station given; quiting ...")
            sys.exit(0)

    # load configuration file
    if args.config:
        loadConfig(args.config)

    # parse mom/out files for sites
    ts = []
    mts = []
    for sta in args.station:
        ts.append(HectorTs.fromMom(sta, args.data_dir))
        if args.plot_model:
            mts.append(HectorTsModel.fromMom(sta, args.data_dir, *ts[-1].breaks()))
            mts[-1].setStartEpoch(ts[-1].startEpoch())

    tmin = min([c.tLimits()[0] for c in ts])
    if args.tmin > tmin:
        tmin = args.tmin
    tmax = max([c.tLimits()[1] for c in ts])
    if args.tmax < tmax:
        tmax = args.tmax

    if args.style_sheet is not None:
        plt.style.use(args.style_sheet)

    fig, ax = plt.subplots(3, 1, sharex=True)

    # remove space between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    # add scatter plots / data points
    sc = []
    for sta in ts:
        ts2 = sta.filter(tmin, tmax)
        ax[0].scatter(ts2.north.t, ts2.north.x, s=plotOptions["markerSize"]["value"])
        ax[1].scatter(ts2.east.t, ts2.east.x, s=plotOptions["markerSize"]["value"])
        sci = ax[2].scatter(
            ts2.up.t, ts2.up.x, s=plotOptions["markerSize"]["value"], label=ts2.site
        )
        sc.append(sci)

    # add model line(s)
    if args.plot_model:
        for idx, msta in enumerate(mts):
            mt, mv = msta.values("north", tmin, tmax)
            ax[0].plot(
                mt,
                mv,
                # c=sc[idx].get_facecolor(),
                linewidth=plotOptions["modelLineWidth"]["value"],
            )
            _, mv = msta.values("east", tmin, tmax)
            ax[1].plot(
                mt,
                mv,
                # c=sc.to_rgba(sc[idx]),
                linewidth=plotOptions["modelLineWidth"]["value"],
            )
            _, mv = msta.values("up", tmin, tmax)
            ax[2].plot(
                mt,
                mv,
                # c=sc.to_rgba(sc[idx]),
                linewidth=plotOptions["modelLineWidth"]["value"],
            )

    # add breaks and jumps
    if not args.no_breaks:
        for sta in ts:
            ts2 = sta.filter(tmin, tmax)
            for brk in ts2.north.breaks:
                ax[0].axvline(brk, linestyle="-.", color="k", lw=0.2)
            for brk in ts2.east.breaks:
                ax[1].axvline(brk, linestyle="-.", color="k", lw=0.2)
            for brk in ts2.up.breaks:
                ax[2].axvline(brk, linestyle="-.", color="k", lw=0.2)

    if len(ts) > 1:
        lgnd = ax[2].legend(
            loc=plotOptions["legendPosition"]["value"],
            fontsize=plotOptions["legendFontSize"]["value"],
        )
        for handle in lgnd.legend_handles:
            handle.set_sizes([plotOptions["legendMarkerSize"]["value"]])

    ax[2].set_xlabel(
        "Time",
        {
            "fontweight": "normal",
            "fontsize": plotOptions["xTitleSize"]["value"],
        },
    )
    ax[0].set_ylabel(
        "North [mm]",
        {
            "fontweight": "normal",
            "fontsize": plotOptions["yTitleSize"]["value"],
        },
    )
    ax[1].set_ylabel(
        "East [mm]",
        {
            "fontweight": "normal",
            "fontsize": plotOptions["yTitleSize"]["value"],
        },
    )
    ax[2].set_ylabel(
        "Up [mm]",
        {
            "fontweight": "normal",
            "fontsize": plotOptions["yTitleSize"]["value"],
        },
    )

    ax[0].set_title(
        "{:} Time Series".format(" ".join(args.station)),
        {
            "fontweight": "heavy",
            "fontsize": plotOptions["titleSize"]["value"],
        },
    )

    # Major formatter/locator (every year)
    ax[0].xaxis.set_major_locator(
        mdates.YearLocator(plotOptions["everyYears"]["value"], month=1, day=1)
    )
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Minor formatter/locator (every n months)
    ax[0].xaxis.set_minor_locator(
        mdates.MonthLocator(range(1, 13, plotOptions["everyMonths"]["value"]), 1)
    )

    fig.autofmt_xdate()
    plt.show()
