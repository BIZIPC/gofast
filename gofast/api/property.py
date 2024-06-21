# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
:code:`gofast` property objects. It is composed of base classes that are inherited 
by methods implemented throughout the package. It also inferred properties to 
data objects. 

.. _GoFast: https://github.com/WEgeophysics/gofast/ 
.. _interpol_imshow: https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html

"""   

# import warnings 
from __future__ import annotations 
# import os 
from abc import ( 
    ABC, 
    abstractmethod, 
    )
import numpy as np
import pandas as pd 
from typing import Any, Dict, Iterable


__all__ = [ 
    "BasePlot", 
    "Config", 
    "UTM_DESIGNATOR",  
    "Software", 
    "Copyright", 
    "References", 
    "Person", 
]


UTM_DESIGNATOR ={
    'X':[72,84], 
    'W':[64,72], 
    'V':[56,64],
    'U':[48,56],
    'T':[40,48],
    'S':[32,40], 
    'R':[24,32], 
    'Q':[16,24], 
    'P':[8,16],
    'N':[0,8], 
    'M':[-8,0],
    'L':[-16, 8], 
    'K':[-24,-16],
    'J':[-32,-24],
    'H':[-40,-32],
    'G':[-48,-40], 
    'F':[-56,-48],
    'E':[-64, -56],
    'D':[-72,-64], 
    'C':[-80,-72],
    'Z':[-80,84]
}
    

class GofastConfig:
    def __init__(self):
        # Initialize the whitespace escape character, setting it as a private attribute
        self._whitespace_escape = "π"

    @property
    def WHITESPACE_ESCAPE(self):
        """
        Gets the whitespace escape character used in the Gofast package for DataFrame
        and API formatting where column names, index names, or values contain whitespaces.

        Returns
        -------
        str
            The character used to escape whitespace in the Gofast package.
        
        Examples
        --------
        >>> config = GofastConfig()
        >>> print(config.WHITESPACE_ESCAPE)
        π

        Notes
        -----
        This property is read-only to prevent changes that could disrupt the
        functionality of the Gofast API frame formatter across all modules.
        """
        return self._whitespace_escape

    @WHITESPACE_ESCAPE.setter
    def WHITESPACE_ESCAPE(self, value):
        """
        Attempt to set the WHITESPACE_ESCAPE property will raise an error.
        
        Raises
        ------
        AttributeError
            If there's an attempt to modify the whitespace escape character.
        """
        raise AttributeError(
            "Modification of WHITESPACE_ESCAPE is not allowed as"
            " it may affect the Gofast API frame formatter across all modules.")

class BaseClass:
    """
    A base class that provides a nicely formatted string representation
    of any derived class instances, summarizing their attributes and
    handling collections intelligently.
    
    Attributes
    ----------
    MAX_DISPLAY_ITEMS : int
        The maximum number of items to display when summarizing
        collections. Default is 5.

    Methods
    -------
    __repr__() -> str
        Returns a formatted representation representation of the instance.
    _format_attr(key: str, value: Any) -> str
        Formats an individual attribute for the string representation.
    _summarize_iterable(iterable: Iterable) -> str
        Returns a summarized string representation of an iterable.
    _summarize_dict(dictionary: Dict) -> str
        Returns a summarized string representation of a dictionary.
    _summarize_array(array: np.ndarray) -> str
        Summarizes a NumPy array to a concise representation.
    _summarize_dataframe(df: pd.DataFrame) -> str
        Summarizes a pandas DataFrame to a concise representation.
    _summarize_series(series: pd.Series) -> str
        Summarizes a pandas Series to a concise representation.
    Examples
    --------
    >>> class Optimizer(BaseClass):
    ...     def __init__(self, name, iterations):
    ...         self.name = name
    ...         self.iterations = iterations
    ...         self.parameters = [1, 2, 3, 4, 5, 6, 7]
    >>> optimizer = Optimizer("SGD", 100)
    >>> print(optimizer)
    Optimizer(name=SGD, iterations=100, parameters=[1, 2, 3, 4, 5, ...])

    Notes
    -----
    The base class is designed to be inherited by classes that wish to
    have a robust and informative string representation for debugging
    and logging purposes. It handles complex data types by summarizing
    them if they exceed the MAX_DISPLAY_ITEMS limit. This behavior can
    be adjusted by modifying the class attribute MAX_DISPLAY_ITEMS.
    """
    MAX_DISPLAY_ITEMS = 5

    def __repr__(self) -> str:
        attributes = [self._format_attr(key, value) 
                      for key, value in self.__dict__.items() 
                      if not key.startswith('_') and not key.endswith('_')
                      ]
        return f"{self.__class__.__name__}({', '.join(attributes)})"

    def _format_attr(self, key: str, value: Any) -> str:
        """Formats a single attribute for inclusion in the string representation."""
        if isinstance(value, (list, tuple, set)):
            return f"{key}={self._summarize_iterable(value)}"
        elif isinstance(value, dict):
            return f"{key}={self._summarize_dict(value)}"
        elif isinstance(value, np.ndarray):
            return f"{key}={self._summarize_array(value)}"
        elif isinstance(value, pd.DataFrame):
            return f"{key}={self._summarize_dataframe(value)}"
        elif isinstance(value, pd.Series):
            return f"{key}={self._summarize_series(value)}"
        else:
            return f"{key}={value}"

    def _summarize_iterable(self, iterable: Iterable) -> str:
        """Summarizes an iterable object to a concise representation 
        if it exceeds the display limit."""
        if len(iterable) > self.MAX_DISPLAY_ITEMS:
            limited_items = ', '.join(map(str, list(iterable)[:self.MAX_DISPLAY_ITEMS]))
            return f"[{limited_items}, ...]"
        else:
            return f"[{', '.join(map(str, iterable))}]"

    def _summarize_dict(self, dictionary: Dict) -> str:
        """Summarizes a dictionary object to a concise representation
        if it exceeds the display limit."""
        if len(dictionary) > self.MAX_DISPLAY_ITEMS:
            limited_items = ', '.join(f"{k}: {v}" for k, v in list(
                dictionary.items())[:self.MAX_DISPLAY_ITEMS])
            return f"{{ {limited_items}, ... }}"
        else:
            return f"{{ {', '.join(f'{k}: {v}' for k, v in dictionary.items()) }}}"

    def _summarize_array(self, array: np.ndarray) -> str:
        """Summarizes array object to a concise representation if 
        it exceeds the display rows."""
        if array.size > self.MAX_DISPLAY_ITEMS:
            limited_items = ', '.join(map(str, array.flatten()[:self.MAX_DISPLAY_ITEMS]))
            return f"[{limited_items}, ...]"
        else:
            return f"[{', '.join(map(str, array.flatten()))}]"

    def _summarize_dataframe(self, df: pd.DataFrame) -> str:
        """Summarizes a DataFrame object to a concise representation if it
        exceeds the display limit."""
        if len(df) > self.MAX_DISPLAY_ITEMS:
            return f"DataFrame({len(df)} rows, {len(df.columns)} columns)"
        else:
            return f"DataFrame: {df.to_string(index=False)}"

    def _summarize_series(self, series: pd.Series) -> str:
        """Summarizes a series object to a concise representation if it
        exceeds the display limit."""
        if len(series) > self.MAX_DISPLAY_ITEMS:
            limited_items = ', '.join(f"{series.index[i]}: {series[i]}" for i in range(
                self.MAX_DISPLAY_ITEMS))
            return f"Series([{limited_items}, ...])"
        else:
            return f"Series: {series.to_string(index=False)}"

class BasePlot(ABC): 
    r""" Base class  deals with Machine learning and conventional Plots. 
    
    The `BasePlot` can not be instanciated. It is build on the top of other 
    plotting classes  and its attributes are used for external plots.
    
    Hold others optional informations: 
        
    ==================  =======================================================
    Property            Description        
    ==================  =======================================================
    fig_dpi             dots-per-inch resolution of the figure
                        *default* is 300
    fig_num             number of the figure instance. *default* is ``1``
    fig_aspect          ['equal'| 'auto'] or float, figure aspect. Can be 
                        rcParams["image.aspect"]. *default* is ``auto``.
    fig_size            size of figure in inches (width, height)
                        *default* is [5, 5]
    savefig             savefigure's name, *default* is ``None``
    fig_orientation     figure orientation. *default* is ``landscape``
    fig_title           figure title. *default* is ``None``
    fs                  size of font of axis tick labels, axis labels are
                        fs+2. *default* is 6 
    ls                  [ '-' | '.' | ':' ] line style of mesh lines
                        *default* is '-'
    lc                  line color of the plot, *default* is ``k``
    lw                  line weight of the plot, *default* is ``1.5``
    alpha               transparency number, *default* is ``0.5``  
    font_weight         weight of the font , *default* is ``bold``.        
    ms                  size of marker in points. *default* is 5
    marker              style  of marker in points. *default* is ``o``.
    marker_facecolor    facecolor of the marker. *default* is ``yellow``
    marker_edgecolor    edgecolor of the marker. *default* is ``cyan``.
    marker_edgewidth    width of the marker. *default* is ``3``.
    xminorticks         minortick according to x-axis size and *default* is 1.
    yminorticks         minortick according to y-axis size and *default* is 1.
    font_size           size of font in inches (width, height)
                        *default* is 3.
    font_style          style of font. *default* is ``italic``
    bins                histograms element separation between two bar. 
                         *default* is ``10``. 
    xlim                limit of x-axis in plot. *default* is None 
    ylim                limit of y-axis in plot. *default* is None 
    xlabel              label name of x-axis in plot. *default* is None 
    ylabel              label name  of y-axis in plot. *default* is None 
    rotate_xlabel       angle to rotate `xlabel` in plot. *default* is None 
    rotate_ylabel       angle to rotate `ylabel` in plot. *default* is None 
    leg_kws             keyword arguments of legend. *default* is empty dict.
    plt_kws             keyword arguments of plot. *default* is empty dict
    plt_style           keyword argument of 2d style. *default* is ``pcolormesh``
    plt_shading         keyword argument of Axes pycolormesh shading. It can be 
                        ['flat'|'nearest'|'gouraud'|'auto'].*default* is 
                        'auto'
    imshow_interp       ['bicubic'|'nearest'|'bilinear'|'quadractic' ] kind of 
                        interpolation for 'imshow' plot. Click `interpol_imshow`_ 
                        to get furher details about the interpolation method. 
                        *default* is ``None``.
    rs                  [ '-' | '.' | ':' ] line style of `Recall` metric
                        *default* is '--'
    ps                  [ '-' | '.' | ':' ] line style of `Precision `metric
                        *default* is '-'
    rc                  line color of `Recall` metric *default* is ``(.6,.6,.6)``
    pc                  line color of `Precision` metric *default* is ``k``
    s                   size of items in scattering plots. default is ``fs*40.``
    cmap                matplotlib colormap. *default* is `jet_r`
    gls                 [ '-' | '.' | ':' ] line style of grid  
                        *default* is '--'.
    glc                 line color of the grid plot, *default* is ``k``
    glw                 line weight of the grid plot, *default* is ``2``
    galpha              transparency number of grid, *default* is ``0.5``  
    gaxis               axis to plot grid.*default* is ``'both'``
    gwhich              type of grid to plot. *default* is ``major``
    tp_axis             axis  to apply ticks params. default is ``both``
    tp_labelsize        labelsize of ticks params. *default* is ``italic``
    tp_bottom           position at bottom of ticks params. *default*
                        is ``True``.
    tp_top              position at the top  of ticks params. *default*
                        is ``True``.
    tp_labelbottom      see label on the bottom of the ticks. *default* 
                        is ``False``
    tp_labeltop         see the label on the top of ticks. *default* is ``True``
    cb_orientation      orientation of the colorbar. *default* is ``vertical``
    cb_aspect           aspect of the colorbar. *default* is 20.
    cb_shrink           shrink size of the colorbar. *default* is ``1.0``
    cb_pad              pad of the colorbar of plot. *default* is ``.05``
    cb_anchor           anchor of the colorbar. *default* is ``(0.0, 0.5)``
    cb_panchor          proportionality anchor of the colorbar. *default* is 
                        `` (1.0, 0.5)``.
    cb_label            label of the colorbar. *default* is ``None``.      
    cb_spacing          spacing of the colorbar. *default* is ``uniform``
    cb_drawedges        draw edges inside of the colorbar. *default* is ``False``
    cb_format           format of the colorbar values. *default* is ``None``.
    sns_orient          seaborn fig orientation. *default* is ``v`` which refer
                        to vertical 
    sns_style           seaborn style 
    sns_palette         seaborn palette 
    sns_height          seaborn height of figure. *default* is ``4.``. 
    sns_aspect          seaborn aspect of the figure. *default* is ``.7``
    sns_theme_kws       seaborn keywords theme arguments. default is ``{
                        'style':4., 'palette':.7}``
    verbose             control the verbosity. Higher value, more messages.
                        *default* is ``0``.
    ==================  =======================================================
    
    """
    
    @abstractmethod 
    def __init__(self,
                 savefig: str = None,
                 fig_num: int =  1,
                 fig_size: tuple =  (12, 8),
                 fig_dpi:int = 300, 
                 fig_legend: str =  None,
                 fig_orientation: str ='landscape',
                 fig_title:str = None,
                 fig_aspect:str='auto',
                 font_size: float =3.,
                 font_style: str ='italic',
                 font_weight: str = 'bold',
                 fs: float = 5.,
                 ms: float =3.,
                 marker: str = 'o',
                 markerfacecolor: str ='yellow',
                 markeredgecolor: str = 'cyan',
                 markeredgewidth: float =  3.,
                 lc: str =  'k',
                 ls: str = '-',
                 lw: float = 1.,
                 alpha: float =  .5,
                 bins: int =  10,
                 xlim: list = None, 
                 ylim: list= None,
                 xminorticks: int=1, 
                 yminorticks: int =1,
                 xlabel: str  =  None,
                 ylabel: str = None,
                 rotate_xlabel: int = None,
                 rotate_ylabel: int =None ,
                 leg_kws: dict = dict(),
                 plt_kws: dict = dict(), 
                 plt_style:str="pcolormesh",
                 plt_shading: str="auto", 
                 imshow_interp:str =None,
                 s: float=  40.,
                 cmap:str='jet_r',
                 show_grid: bool = False,
                 galpha: float = .5,
                 gaxis: str = 'both',
                 gc: str = 'k',
                 gls: str = '--',
                 glw: float = 2.,
                 gwhich: str = 'major',               
                 tp_axis: str = 'both',
                 tp_labelsize: float = 3.,
                 tp_bottom: bool =True,
                 tp_top: bool = True,
                 tp_labelbottom: bool = False,
                 tp_labeltop: bool = True,               
                 cb_orientation: str = 'vertical',
                 cb_aspect: float = 20.,
                 cb_shrink: float =  1.,
                 cb_pad: float =.05,
                 cb_anchor: tuple = (0., .5),
                 cb_panchor: tuple=  (1., .5),              
                 cb_label: str = None,
                 cb_spacing: str = 'uniform' ,
                 cb_drawedges: bool = False,
                 cb_format: float = None ,   
                 sns_orient: str ='v', 
                 sns_style: str = None, 
                 sns_palette: str= None, 
                 sns_height: float=4. , 
                 sns_aspect:float =.7, 
                 sns_theme_kws: dict = None,
                 verbose: int=0, 
                 ): 
        
        self.savefig=savefig
        self.fig_num=fig_num
        self.fig_size=fig_size
        self.fig_dpi=fig_dpi
        self.fig_legend=fig_legend
        self.fig_orientation=fig_orientation
        self.fig_title=fig_title
        self.fig_aspect=fig_aspect
        self.font_size=font_size
        self.font_style=font_style
        self.font_weight=font_weight
        self.fs=fs
        self.ms=ms
        self.marker=marker
        self.marker_facecolor=markerfacecolor
        self.marker_edgecolor=markeredgecolor
        self.marker_edgewidth=markeredgewidth
        self.lc=lc
        self.ls=ls
        self.lw=lw
        self.alpha=alpha
        self.bins=bins
        self.xlim=xlim
        self.ylim=ylim
        self.x_minorticks=xminorticks
        self.y_minorticks=yminorticks
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.rotate_xlabel=rotate_xlabel
        self.rotate_ylabel=rotate_ylabel
        self.leg_kws=leg_kws
        self.plt_kws=plt_kws
        self.plt_style=plt_style
        self.plt_shading=plt_shading
        self.imshow_interp=imshow_interp
        self.s=s 
        self.cmap=cmap
        self.show_grid=show_grid
        self.galpha=galpha
        self.gaxis=gaxis
        self.gc=gc
        self.gls=gls
        self.glw=glw
        self.gwhich=gwhich
        self.tp_axis=tp_axis
        self.tp_labelsize=tp_labelsize  
        self.tp_bottom=tp_bottom
        self.tp_top=tp_top
        self.tp_labelbottom=tp_labelbottom
        self.tp_labeltop=tp_labeltop
        self.cb_orientation=cb_orientation
        self.cb_aspect=cb_aspect
        self.cb_shrink=cb_shrink
        self.cb_pad=cb_pad
        self.cb_anchor=cb_anchor
        self.cb_panchor=cb_panchor
        self.cb_label=cb_label
        self.cb_spacing=cb_spacing
        self.cb_drawedges=cb_drawedges
        self.cb_format=cb_format  
        self.sns_orient=sns_orient
        self.sns_style=sns_style
        self.sns_palette=sns_palette
        self.sns_height=sns_height
        self.sns_aspect=sns_aspect
        self.verbose=verbose
        self.sns_theme_kws=sns_theme_kws or {'style':self.sns_style, 
                                         'palette':self.sns_palette, 
                                                      }
        self.cb_props = {
            pname.replace('cb_', '') : pvalues
                         for pname, pvalues in self.__dict__.items() 
                         if pname.startswith('cb_')
                         }
       
         
    
class Config: 
    
    """ Container of property elements. 
    
    Out of bag to keep unmodificable elements. Trick to encapsulate all the 
    element that are not be allow to be modified.
    
    """
    
    @property 
    def arraytype (self):
        """ Different array from |ERP| configuration. 
    
         Array-configuration  can be added as the development progresses. 
         
        """
        return {
        1 : (
            ['Schlumberger','AB>> MN','slbg'], 
            'S'
            ), 
        2 : (
            ['Wenner','AB=MN'], 
             'W'
             ), 
        3: (
            ['Dipole-dipole','dd','AB<BM>MN','MN<NA>AB'],
            'DD'
            ), 
        4: (
            ['Gradient-rectangular','[AB]MN', 'MN[AB]','[AB]'],
            'GR'
            )
        }
    @property
    def parsers(self ): 
        """ Readable format that can be read and parse the data  """
        return {
                 ".csv" : pd.read_csv, 
                 ".xlsx": pd.read_excel,
                 ".json": pd.read_json,
                 ".html": pd.read_html,
                 ".sql" : pd.read_sql, 
                 ".xml" : pd.read_xml , 
                 ".fwf" : pd.read_fwf, 
                 ".pkl" : pd.read_pickle, 
                 ".sas" : pd.read_sas, 
                 ".spss": pd.read_spss,
                 # ".orc" : pd.read_orc, 
                 }
        
    @staticmethod
    def writers (object): 
        """ Write frame formats."""
        return {".csv"    : object.to_csv, 
                ".hdf"    : object.to_hdf, 
                ".sql"    : object.to_sql, 
                ".dict"   : object.to_dict, 
                ".xlsx"   : object.to_excel, 
                ".json"   : object.to_json, 
                ".html"   : object.to_html , 
                ".feather": object.to_feather, 
                ".tex"    : object.to_latex, 
                ".stata"  : object.to_stata, 
                ".gbq"    : object.to_gbq, 
                ".rec"    : object.to_records, 
                ".str"    : object.to_string, 
                ".clip"   : object.to_clipboard, 
                ".md"     : object.to_markdown, 
                ".parq"   : object.to_parquet, 
                # ".orc"    : object.to_orc, 
                ".pkl"    : object.to_pickle 
                }
    
    @staticmethod 
    def arrangement(a: int | str ): 
        """ Assert whether the given arrangement is correct. 
        
        :param a: int, float, str - Type of given electrical arrangement. 
        
        :returns:
            - The correct arrangement name 
            - ``0`` which means ``False`` or a wrong given arrangements.   
        """
        
        for k, v in Config().arraytype.items(): 
            if a == k  or str(a).lower().strip() in ','.join (
                    v[0]).lower() or a ==v[1]: 
                return  v[0][0].lower()
            
        return 0
    
    @property 
    def geo_rocks_properties(self ):
        """ Get some sample of the geological rocks. """
        return {
             "hard rock" :                 [1e99,1e6 ],
             "igneous rock":               [1e6, 1e3], 
             "duricrust"   :               [5.1e3 , 5.1e2],
             "gravel/sand" :               [1e4  , 7.943e0],
             "conglomerate"    :           [1e4  , 8.913e1],
             "dolomite/limestone" :        [1e5 ,  1e3],
             "permafrost"  :               [1e5  , 4.169e2],
             "metamorphic rock" :          [5.1e2 , 1e1],
             "tills"  :                    [8.1e2 , 8.512e1],
             "standstone conglomerate" :   [1e4 , 8.318e1],
             "lignite/coal":               [7.762e2 , 1e1],
             "shale"   :                   [5.012e1 , 3.20e1],
             "clay"   :                    [1e2 ,  5.012e1],
             "saprolite" :                 [6.310e2 , 3.020e1],
             "sedimentary rock":           [1e4 , 1e0],
             "fresh water"  :              [3.1e2 ,1e0],
             "salt water"   :              [1e0 , 1.41e0],
             "massive sulphide" :          [1e0   ,  1e-2],
             "sea water"     :             [1.231e-1 ,1e-1],
             "ore minerals"  :             [1e0   , 1e-4],
             "graphite"    :               [3.1623e-2, 3.162e-3]
                
                }
    
    @property 
    def rockpatterns(self): 
        """Default geological rocks patterns. 
        
        pattern are not exhaustiv, can be added and changed. This pattern
        randomly choosen its not exatly match the rocks geological patterns 
        as described with the conventional geological swatches relate to 
        the USGS(US Geological Survey ) swatches- references and FGDC 
        (Digital cartographic Standard for Geological  Map Symbolisation 
         -FGDCgeostdTM11A2_A-37-01cs2.eps)
        
        The following symbols can be used to create a matplotlib pattern. 
        
        .. code-block:: none
        
            make _pattern:{'/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
                           
                /   - diagonal hatching
                \\   - back diagonal
                |   - vertical
                -   - horizontal
                +   - crossed
                x   - crossed diagonal
                o   - small circle
                O   - large circle
                .   - dots
                *   - stars
        """
        return  {
             "hard rock" :           ['.+++++.', (.25, .5, .5)],
             "igneous rock":         ['.o.o.', (1., 1., 1.)], 
             "duricrust"   :         ['+.+',(1., .2, .36)],
             "gravel" :              ['oO',(.75,.86,.12)],
             "sand":                 ['....',(.23, .36, .45)],
             "conglomerate"    :     ['.O.', (.55, 0., .36)],
             "dolomite" :            ['.-.', (0., .75, .23)],
             "limestone" :           ['//.',(.52, .23, .125)],
            "permafrost"  :          ['o.', (.2, .26, .75)],
             "metamorphic rock" :    ['*o.', (.2, .2, .3)],
             "tills"  :              ['-.', (.7, .6, .9)],
             "standstone ":          ['..', (.5, .6, .9)],
             "lignite coal":         ['+/.',(.5, .5, .4)],
             "coal":                 ['*.', (.8, .9, 0.)],
             "shale"   :             ['=', (0., 0., 0.7)],
             "clay"   :              ['=.',(.9, .8, 0.8)],
             "saprolite" :           ['*/',(.3, 1.2, .4)],
             "sedimentary rock":     ['...',(.25, 0., .25)],
             "fresh water"  :        ['.-.',(0., 1.,.2)],
             "salt water"   :        ['o.-',(.2, 1., .2)],
             "massive sulphide" :    ['.+O',(1.,.5, .5 )],
             "sea water"     :       ['.--',(.0, 1., 0.)],
             "ore minerals"  :       ['--|',(.8, .2, .2)],
             "graphite"    :         ['.++.',(.2, .7, .7)],
             
             }
    
class References:
    """
    References information for a citation.

    Holds the following information:
        
    ================  ==========  =============================================
    Attributes         Type        Explanation
    ================  ==========  =============================================
    author            string      Author names
    title             string      Title of article, or publication
    journal           string      Name of journal
    doi               string      DOI number 
    year              int         year published
    ================  ==========  =============================================

    More attributes can be added by inputing a key word dictionary
    
    Examples
    ---------
    >>> from gofast.property import References
    >>> refobj = References(
        **{'volume':18, 'pages':'234--214', 
        'title':'gofast :A machine learning research for hydrogeophysic' ,
        'journal':'Computers and Geosciences', 
        'year':'2021', 'author':'DMaryE'}
        )
    >>> refobj.journal
    Out[21]: 'Computers and Geosciences'
    """
    def __init__(
        self, 
        author=None, 
        title=None, 
        journal=None, 
        volume=None, 
        doi=None, 
        year=None,  
        **kws
        ):
        self.author=author 
        self.title=title 
        self.journal=journal 
        self.volume=volume 
        self.doi=doi 
        self.year=year 
   
        for key in list(kws.keys()):
            setattr(self, key, kws[key])


class Copyright:
    """
    Information of copyright, mainly about the use of data can use
    the data. Be sure to read over the conditions_of_use.

    Holds the following informations:

    =================  ===========  ===========================================
    Attributes         Type         Explanation
    =================  ===========  ===========================================
    References          References  citation of published work using these data
    conditions_of_use   string      conditions of use of data used for testing 
                                    program
    release_status      string      release status [ open | public |proprietary]
    =================  ===========  ===========================================

    More attributes can be added by inputing a key word dictionary
    
    Examples
    ----------
    >>> from gofast.property import Copyright 
    >>> copbj =Copyright(**{'owner':'University of AI applications',
    ...             'contact':'WATER4ALL'})
    >>> copbj.contact 
    Out[20]: 'WATER4ALL
    
    """
    cuse =( 
        "All Data used for software demonstration mostly located in "
        " data directory <data/> cannot be used for commercial and " 
        " distributive purposes. They can not be distributed to a third"
        " party. However, they can be used for understanding the program."
        " Some available ERP and VES raw data can be found on the record"
        " <'10.5281/zenodo.5571534'>. Whereas EDI-data e.g. EMAP/MT data,"
        " can be collected at http://ds.iris.edu/ds/tags/magnetotelluric-data/."
        " The metadata from both sites are available free of charge and may"
        " be copied freely, duplicated and further distributed provided"
        " these data are cited as the reference."
        )
    def __init__(
        self, 
        release_status=None, 
        additional_info=None, 
        conditions_of_use=None, 
        **kws
        ):
        self.release_status=release_status
        self.additional_info=additional_info
        self.conditions_of_use=conditions_of_use or self.cuse 
        self.References=References()
        for key in list(kws.keys()):
            setattr(self, key, kws[key])


class Person:
    """
    Information for a person

    ================  ==========  =============================================
    Attributes         Type        Explanation
    ================  ==========  =============================================
    email             string      email of person
    name              string      name of person
    organization      string      name of person's organization
    organization_url  string      organizations web address
    ================  ==========  =============================================

    More attributes can be added by inputing a key word dictionary
    
    Examples 
    ----------
    >>> from gofast.property import Person
    >>> person =Person(**{'name':'ABA', 'email':'aba@water4all.ai.org',
    ...                  'phone':'00225-0769980706', 
    ...          'organization':'WATER4ALL'})
    >>> person.name
    Out[23]: 'ABA
    >>> person.organization
    Out[25]: 'WATER4ALL'
    """

    def __init__(
        self, 
        email=None, 
        name=None, 
        organization=None, 
        organization_url=None, 
        **kws
        ):
        self.email=email 
        self.name=name 
        self.organization=organization
        self.organization_url=organization_url

        for key in list(kws.keys()):
            setattr(self, key, kws[key])


class Software:
    """
    software info 

    ================= =========== =============================================
    Attributes         Type        Explanation
    ================= =========== =============================================
    name                string      name of software 
    version             string      version of sotware 
    Author              string      Author of software
    release             string      latest version release
    ================= =========== =============================================
    
    More attributes can be added by inputing a key word dictionary

    Examples 
    ----------
    >>> from gofast.property import Software
    >>> Software(**{'release':'0.11.23'})

    """
    def __init__(
        self,
        name=None, 
        version=None, 
        release=None, 
        **kws
        ):
        self.name=name 
        self.version=version 
        self.release=release 
        self.Author=Person()
        
        for key in kws:
            setattr(self, key, kws[key]) 
            
                

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   