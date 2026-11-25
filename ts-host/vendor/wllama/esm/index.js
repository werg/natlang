var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  }, method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no)), it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var __forAwait = (obj, it, method) => (it = obj[__knownSymbol("asyncIterator")]) ? it.call(obj) : (obj = obj[__knownSymbol("iterator")](), it = {}, method = (key, fn) => (fn = obj[key]) && (it[key] = (arg) => new Promise((yes, no, done) => (arg = fn.call(obj, arg), done = arg.done, Promise.resolve(arg.value).then((value) => yes({ value, done }), no)))), method("next"), method("return"), it);

// src/glue/messages.ts
var GLUE_VERSION = 1;
var GLUE_MESSAGE_PROTOTYPES = {
  "erro_evt": {
    "name": "erro_evt",
    "structName": "glue_msg_error",
    "className": "GlueMsgError",
    "fields": [
      {
        "type": "str",
        "name": "message",
        "isNullable": false
      }
    ]
  },
  "load_req": {
    "name": "load_req",
    "structName": "glue_msg_load_req",
    "className": "GlueMsgLoadReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "model_paths",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "mmproj_path",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "n_ctx_auto",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mmap",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mlock",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_gpu_layers",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_threads",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "model_alias",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "log_level",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "embeddings",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "offload_kqv",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_parallel",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "pooling_type",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "rope_scaling_type",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_base",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_scale",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_ext_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_attn_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_fast",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_slow",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "yarn_orig_ctx",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_k",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_v",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "kv_unified",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "flash_attn",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "swa_full",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ctx_checkpoints",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "checkpoint_min_step",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "chat_template",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "jinja",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_vals",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "reasoning",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_min_tokens",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_max_tokens",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "warmup",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "no_kv_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "mmproj_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cont_batching",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_keep",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "ctx_shift",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cache_idle_slots",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_cache_reuse",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "lora_paths",
        "isNullable": true
      },
      {
        "type": "arr_float",
        "name": "lora_scales",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "lora_init_without_apply",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "native_state_paths",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "spec_draft_model",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_ngl",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_max",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_min",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "spec_draft_p_min",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads_batch",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_vals",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "reasoning_budget_tokens",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_budget_message",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_format",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "skip_chat_parsing",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "prefill_assistant",
        "isNullable": true
      }
    ]
  },
  "load_res": {
    "name": "load_res",
    "structName": "glue_msg_load_res",
    "className": "GlueMsgLoadRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_vocab",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx_train",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_embd",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_layer",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_key",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_val",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_bos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eot",
        "isNullable": false
      },
      {
        "type": "arr_int",
        "name": "list_tokens_eog",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_bos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_eos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_encoder",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_decoder_start",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "media_marker",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_image_input",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_audio_input",
        "isNullable": false
      }
    ]
  },
  "cmpl_req": {
    "name": "cmpl_req",
    "structName": "glue_msg_completion_req",
    "className": "GlueMsgCompletionReq",
    "fields": [
      {
        "type": "bool",
        "name": "is_chat",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "cmpl_res": {
    "name": "cmpl_res",
    "structName": "glue_msg_completion_res",
    "className": "GlueMsgCompletionRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "embd_req": {
    "name": "embd_req",
    "structName": "glue_msg_embedding_req",
    "className": "GlueMsgEmbeddingReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "embd_res": {
    "name": "embd_res",
    "structName": "glue_msg_embedding_res",
    "className": "GlueMsgEmbeddingRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "rrnk_req": {
    "name": "rrnk_req",
    "structName": "glue_msg_rerank_req",
    "className": "GlueMsgRerankReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "rrnk_res": {
    "name": "rrnk_res",
    "structName": "glue_msg_rerank_res",
    "className": "GlueMsgRerankRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "sys1_req": {
    "name": "sys1_req",
    "structName": "glue_msg_systemone_req",
    "className": "GlueMsgSystemoneReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "sys1_res": {
    "name": "sys1_res",
    "structName": "glue_msg_systemone_res",
    "className": "GlueMsgSystemoneRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "scor_req": {
    "name": "scor_req",
    "structName": "glue_msg_score_req",
    "className": "GlueMsgScoreReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "scor_res": {
    "name": "scor_res",
    "structName": "glue_msg_score_res",
    "className": "GlueMsgScoreRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "slot_req": {
    "name": "slot_req",
    "structName": "glue_msg_slots_req",
    "className": "GlueMsgSlotsReq",
    "fields": [
      {
        "type": "int",
        "name": "id_slot",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "slot_res": {
    "name": "slot_res",
    "structName": "glue_msg_slots_res",
    "className": "GlueMsgSlotsRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "nsta_req": {
    "name": "nsta_req",
    "structName": "glue_msg_native_state_req",
    "className": "GlueMsgNativeStateReq",
    "fields": [
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "nsta_res": {
    "name": "nsta_res",
    "structName": "glue_msg_native_state_res",
    "className": "GlueMsgNativeStateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "meta_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_req": {
    "name": "tmpl_req",
    "structName": "glue_msg_apply_template_req",
    "className": "GlueMsgApplyTemplateReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_res": {
    "name": "tmpl_res",
    "structName": "glue_msg_apply_template_res",
    "className": "GlueMsgApplyTemplateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "prompt",
        "isNullable": false
      }
    ]
  },
  "gres_req": {
    "name": "gres_req",
    "structName": "glue_msg_get_result_req",
    "className": "GlueMsgGetResultReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_res": {
    "name": "gres_res",
    "structName": "glue_msg_get_result_res",
    "className": "GlueMsgGetResultRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_more",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "is_error",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "cncl_req": {
    "name": "cncl_req",
    "structName": "glue_msg_cancel_req",
    "className": "GlueMsgCancelReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "cncl_res": {
    "name": "cncl_res",
    "structName": "glue_msg_cancel_res",
    "className": "GlueMsgCancelRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  },
  "tbop_req": {
    "name": "tbop_req",
    "structName": "glue_msg_test_backend_ops_req",
    "className": "GlueMsgTestBackendOpsReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "args",
        "isNullable": false
      }
    ]
  },
  "tbop_res": {
    "name": "tbop_res",
    "structName": "glue_msg_test_backend_ops_res",
    "className": "GlueMsgTestBackendOpsRes",
    "fields": [
      {
        "type": "int",
        "name": "retcode",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  }
};

// src/glue/glue.ts
var GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);
var GLUE_DTYPE_NULL = 0;
var GLUE_DTYPE_BOOL = 1;
var GLUE_DTYPE_INT = 2;
var GLUE_DTYPE_FLOAT = 3;
var GLUE_DTYPE_STRING = 4;
var GLUE_DTYPE_RAW = 5;
var GLUE_DTYPE_ARRAY_BOOL = 6;
var GLUE_DTYPE_ARRAY_INT = 7;
var GLUE_DTYPE_ARRAY_FLOAT = 8;
var GLUE_DTYPE_ARRAY_STRING = 9;
var GLUE_DTYPE_ARRAY_RAW = 10;
var TYPE_MAP = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL
};
function glueDeserialize(buf) {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen) => {
    const length = customLen != null ? customLen : readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;
  const readField = (field) => {
    switch (field.type) {
      case "str":
        return readString();
      case "int":
        return readInt32();
      case "float":
        return readFloat();
      case "bool":
        return readBool();
      case "raw":
        return readRaw();
      case "arr_str":
        return readArray(readString);
      case "arr_int":
        return readArray(readInt32);
      case "arr_float":
        return readArray(readFloat);
      case "arr_bool":
        return readArray(readBool);
      case "arr_raw":
        return readArray(readRaw);
      case "null":
        return readNull();
    }
  };
  const magicValid = buf[0] === GLUE_MAGIC[0] && buf[1] === GLUE_MAGIC[1] && buf[2] === GLUE_MAGIC[2] && buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error("Invalid magic number");
  }
  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error("Invalid version number");
  }
  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }
  const output = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }
  return output;
}
function glueSerialize(msg) {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }
  const bufs = [];
  const writeUint32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value, writeItem) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {
  };
  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = msg[field.name];
    if (!field.isNullable && (val === null || val === void 0)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === void 0) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case "str":
        writeString(val);
        break;
      case "int":
        writeInt32(val);
        break;
      case "float":
        writeFloat(val);
        break;
      case "bool":
        writeBool(val);
        break;
      case "raw":
        writeRaw(val);
        break;
      case "arr_str":
        writeArray(val, writeString);
        break;
      case "arr_int":
        writeArray(val, writeInt32);
        break;
      case "arr_float":
        writeArray(val, writeFloat);
        break;
      case "arr_bool":
        writeArray(val, writeBool);
        break;
      case "arr_raw":
        writeArray(val, writeRaw);
        break;
      case "null":
        writeNull();
        break;
    }
  }
  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}

// src/wasm/source-map.ts
var WASM_SOURCE_MAP = {
  "default": "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",
  "compat": "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"
};

// src/debug.ts
var cache = /* @__PURE__ */ new Map();
function loadMap(buildKey) {
  return __async(this, null, function* () {
    if (cache.has(buildKey)) return cache.get(buildKey);
    const b64 = WASM_SOURCE_MAP[buildKey];
    if (!b64) throw new Error(`No source map for build "${buildKey}"`);
    const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    writer.write(gzipped);
    writer.close();
    const buf = yield new Response(ds.readable).arrayBuffer();
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const firstId = dv.getUint32(0, true);
    const funcCount = dv.getUint32(4, true);
    const numNames = dv.getUint32(8, true);
    const td = new TextDecoder();
    const names = [];
    let pos = 12;
    for (let i = 0; i < numNames; i++) {
      const len = bytes[pos++];
      names.push(td.decode(bytes.subarray(pos, pos + len)));
      pos += len;
    }
    const funcNames = [];
    for (let i = 0; i < funcCount; i++) {
      const idx = dv.getUint16(pos, true);
      pos += 2;
      funcNames.push(idx === 65535 ? null : names[idx]);
    }
    const entry = { firstId, funcNames };
    cache.set(buildKey, entry);
    return entry;
  });
}
var Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: (funcIds, isCompatBuild) => __async(void 0, null, function* () {
    const buildKey = isCompatBuild ? "compat" : "default";
    const { firstId, funcNames } = yield loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name = i >= 0 && i < funcNames.length && funcNames[i] ? funcNames[i] : "(unknown)";
      return { funcId, name };
    });
  }),
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: (stack, isCompatBuild) => __async(void 0, null, function* () {
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1])))
    ];
    if (funcIds.length === 0) return stack;
    const resolved = yield Debug.decodeFuncIds(funcIds, isCompatBuild);
    return resolved.map((r) => {
      if (r.name === "(unknown)") {
        return `    wasm-func[${r.funcId}] (unknown)`;
      }
      return `    wasm-func[${r.funcId}] (${r.name})`;
    }).join("\n");
  })
};

// src/utils.ts
var textDecoder = new TextDecoder();
var URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
var parseShardNumber = (fnameOrUrl) => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ""),
      current: parseInt(matches[1]),
      total: parseInt(matches[2])
    };
  }
};
var sortFileByShard = (blobs) => {
  const isFiles = blobs.every((b) => !!b.name);
  if (isFiles && blobs.length > 1) {
    const files = blobs;
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
  }
};
var isMmproj = (blob) => __async(void 0, null, function* () {
  const META_NAME = "general.architecture";
  const META_VAL = "clip";
  const tmp = blob.slice(0, 128 * 1024);
  const header = yield tmp.arrayBuffer();
  const buf = new Uint8Array(header);
  const nameBytes = new TextEncoder().encode(META_NAME);
  const valBytes = new TextEncoder().encode(META_VAL);
  let offset = -1;
  outer: for (let i = 0; i <= buf.length - nameBytes.length; i++) {
    for (let j = 0; j < nameBytes.length; j++) {
      if (buf[i + j] !== nameBytes[j]) continue outer;
    }
    offset = i;
    break;
  }
  if (offset === -1) return false;
  if (offset + 8 * 4 + 4 > buf.length) return false;
  const view = new DataView(header);
  const valLen = view.getBigUint64(offset + 8 * 3, true);
  if (valLen !== /* @__PURE__ */ BigInt("4")) return false;
  for (let i = 0; i < valBytes.length; i++) {
    if (buf[offset + 8 * 4 + i] !== valBytes[i]) return false;
  }
  return true;
});
var absoluteUrl = (relativePath) => typeof document === "undefined" ? new URL(relativePath, self.location.href).href : new URL(relativePath, document.baseURI).href;
var padDigits = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join("0") + number;
};
var sumArr = (arr) => arr.reduce((prev, curr) => prev + curr, 0);
var isString = (value) => !!(value == null ? void 0 : value.startsWith);
var MMPROJ_FILE_NAME = "mmproj.gguf";
var prepareBlobs = (blobsInp) => __async(void 0, null, function* () {
  const blobs = [];
  let blobMmproj = null;
  for (const blob of blobsInp) {
    if (yield isMmproj(blob)) {
      blobMmproj = blob;
    } else {
      blobs.push(blob);
    }
  }
  sortFileByShard(blobs);
  const result = blobs.map((blob, i) => ({
    blob,
    name: `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
  }));
  if (blobMmproj) {
    result.push({
      blob: blobMmproj,
      name: MMPROJ_FILE_NAME
    });
  }
  return {
    llm: result.filter((f) => f.name !== MMPROJ_FILE_NAME),
    mmproj: blobMmproj ? { blob: blobMmproj, name: MMPROJ_FILE_NAME } : null,
    all: result
  };
});
var isSupportMultiThread = () => ((e) => __async(void 0, null, function* () {
  try {
    return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);
  } catch (e2) {
    return false;
  }
}))(
  new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    1,
    4,
    1,
    96,
    0,
    0,
    3,
    2,
    1,
    0,
    5,
    4,
    1,
    3,
    1,
    1,
    10,
    11,
    1,
    9,
    0,
    65,
    0,
    254,
    16,
    2,
    0,
    26,
    11
  ])
);
var isSupportExceptions = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      4,
      1,
      96,
      0,
      0,
      3,
      2,
      1,
      0,
      10,
      8,
      1,
      6,
      0,
      6,
      64,
      25,
      11,
      11
    ])
  );
});
var isSupportSIMD = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      5,
      1,
      96,
      0,
      1,
      123,
      3,
      2,
      1,
      0,
      10,
      10,
      1,
      8,
      0,
      65,
      0,
      253,
      15,
      253,
      98,
      11
    ])
  );
});
var isSupportJSPI = () => {
  return !!WebAssembly.Suspending;
};
var isSupportWebGPU = () => {
  return !!navigator.gpu;
};
var isSupportMem64 = () => {
  try {
    new WebAssembly.Memory({
      address: "i64",
      initial: /* @__PURE__ */ BigInt("1")
      // 1 page (64 KiB)
    });
    return true;
  } catch (e) {
    return false;
  }
};
var checkEnvironmentCompatible = () => __async(void 0, null, function* () {
  if (!(yield isSupportExceptions())) {
    throw new Error("WebAssembly runtime does not support exception handling");
  }
  if (!(yield isSupportSIMD())) {
    throw new Error("WebAssembly runtime does not support SIMD");
  }
});
var isFirefox = () => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
var GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;
var isValidGgufFile = (path) => {
  return GGUF_FILE_REGEX.test(path);
};
var isSafariMobile = () => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/);
};
var createWorker = (workerCode) => {
  const workerURL = URL.createObjectURL(
    isString(workerCode) ? new Blob([workerCode], { type: "text/javascript" }) : workerCode
  );
  return new Worker(workerURL, { type: "module" });
};
var cbToAsyncIter = (fn) => (...args) => {
  let values = [];
  let resolve;
  let reject;
  values.push(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    })
  );
  fn(...args, (val, done, err) => {
    if (err) {
      reject(err);
      return;
    }
    resolve([val, done]);
    values.push(
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
    );
  });
  return function() {
    return __asyncGenerator(this, null, function* () {
      let val;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = yield new __await(values[i]);
        delete values[i];
        if (val !== void 0) yield val;
      }
    });
  }();
};
var canUseAsyncFileRead = (compat) => isSupportJSPI() || compat;
var needCompat = () => !isSupportJSPI() || !isSupportMem64();

// src/workers-code/generated.ts
var LIBLLAMA_VERSION = "b12094-b536c73";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n  // slot save files (action \"slots\"), moved to and from the main thread as bytes\n  m.FS.mkdir('/slots');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_read') {\n    const argFilename = args[0];\n    try {\n      const path = '/slots/' + argFilename;\n      const data = Module.FS.readFile(path);\n      Module.FS.unlink(path);\n      msg({ callbackId, result: data }, [data.buffer]);\n    } catch (err) {\n      msg({ callbackId, err: `cannot read slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_write') {\n    const argFilename = args[0];\n    const argData = args[1];\n    try {\n      Module.FS.writeFile('/slots/' + argFilename, argData);\n      msg({ callbackId, result: null });\n    } catch (err) {\n      msg({ callbackId, err: `cannot write slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_remove') {\n    try {\n      Module.FS.unlink('/slots/' + args[0]);\n    } catch (err) {\n      // already removed\n    }\n    msg({ callbackId, result: null });\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
var OPFS_UTILS_WORKER_CODE = "let accessHandle;\nlet abortController = new AbortController();\n\nasync function openFile(filename) {\n  const opfsRoot = await navigator.storage.getDirectory();\n  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });\n  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });\n  accessHandle = await fileHandler.createSyncAccessHandle();\n  accessHandle.truncate(0); // clear file content\n}\n\nasync function writeFile(buf) {\n  accessHandle.write(buf);\n}\n\nasync function closeFile() {\n  accessHandle.flush();\n  accessHandle.close();\n}\n\nasync function writeTextFile(filename, str) {\n  await openFile(filename);\n  await writeFile(new TextEncoder().encode(str));\n  await closeFile();\n}\n\nconst throttled = (func, delay) => {\n  let lastRun = 0;\n  return (...args) => {\n    const now = Date.now();\n    if (now - lastRun > delay) {\n      lastRun = now;\n      func.apply(null, args);\n    }\n  };\n};\n\nconst assertNonNull = (val) => {\n  if (val === null || val === undefined) {\n    throw new Error('OPFS Worker: Assertion failed');\n  }\n};\n\n// respond to main thread\nconst resOK = () => postMessage({ ok: true });\nconst resProgress = (loaded, total) =>\n  postMessage({ progress: { loaded, total } });\nconst resErr = (err) => postMessage({ err });\n\nonmessage = async (e) => {\n  try {\n    if (!e.data) return;\n\n    /**\n     * @param {Object} e.data\n     *\n     * Fine-control FS actions:\n     * - { action: 'open', filename: 'string' }\n     * - { action: 'write', buf: ArrayBuffer }\n     * - { action: 'close' }\n     *\n     * Simple write API:\n     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }\n     *\n     * Download API:\n     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }\n     * - { action: 'download-abort' }\n     */\n    const {\n      action,\n      filename,\n      buf,\n      url,\n      options,\n      metadataFileName,\n      metadataAdditional,\n    } = e.data;\n\n    if (action === 'open') {\n      assertNonNull(filename);\n      await openFile(filename);\n      return resOK();\n    } else if (action === 'write') {\n      assertNonNull(buf);\n      await writeFile(buf);\n      return resOK();\n    } else if (action === 'close') {\n      await closeFile();\n      return resOK();\n    } else if (action === 'write-simple') {\n      assertNonNull(filename);\n      assertNonNull(buf);\n      await openFile(filename);\n      await writeFile(buf);\n      await closeFile();\n      return resOK();\n    } else if (action === 'download') {\n      assertNonNull(url);\n      assertNonNull(filename);\n      assertNonNull(metadataFileName);\n      assertNonNull(options);\n      assertNonNull(options.aborted);\n      abortController = new AbortController();\n      if (options.aborted) abortController.abort();\n      const response = await fetch(url, {\n        ...options,\n        signal: abortController.signal,\n      });\n      const contentLength = response.headers.get('content-length');\n      const etag = (response.headers.get('etag') || '').replace(\n        /[^A-Za-z0-9]/g,\n        ''\n      );\n      const total = parseInt(contentLength, 10);\n      const reader = response.body.getReader();\n      await openFile(filename);\n      let loaded = 0;\n      const throttledProgress = throttled(resProgress, 100);\n      while (true) {\n        const { done, value } = await reader.read();\n        if (done) break;\n        loaded += value.byteLength;\n        await writeFile(value);\n        throttledProgress(loaded, total);\n      }\n      resProgress(total, total); // 100% done\n      await closeFile();\n      // make sure this is in-sync with CacheEntryMetadata\n      await writeTextFile(\n        metadataFileName,\n        JSON.stringify({\n          originalURL: url,\n          originalSize: total,\n          etag,\n          ...metadataAdditional,\n        })\n      );\n      return resOK();\n    } else if (action === 'download-abort') {\n      if (abortController) {\n        abortController.abort();\n      }\n      return;\n    }\n\n    throw new Error('OPFS Worker: Invalid action', e.data);\n  } catch (err) {\n    return resErr(err);\n  }\n};\n";
var WLLAMA_EMSCRIPTEN_CODE = 'var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&self.name?.startsWith("em-pthread");if(ENVIRONMENT_IS_NODE){var worker_threads=require("worker_threads");global.Worker=worker_threads.Worker;ENVIRONMENT_IS_WORKER=!worker_threads.isMainThread;ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&worker_threads["workerData"]=="em-pthread"}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\\\/g,"/")}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}if(!ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var defaultPrint=console.log.bind(console);var defaultPrintErr=console.error.bind(console);if(ENVIRONMENT_IS_NODE){var utils=require("util");var stringify=a=>typeof a=="object"?utils.inspect(a):a;defaultPrint=(...args)=>fs.writeSync(1,args.map(stringify).join(" ")+"\\n");defaultPrintErr=(...args)=>fs.writeSync(2,args.map(stringify).join(" ")+"\\n")}var out=defaultPrint;var err=defaultPrintErr;var wasmBinary;var wasmModule;var ABORT=false;var EXITSTATUS;function assert(condition,text){if(!condition){abort(text)}}var isFileURI=filename=>filename.startsWith("file://");function growMemViews(){if(wasmMemory.buffer!=HEAP8.buffer){updateMemoryViews()}}if(ENVIRONMENT_IS_NODE&&ENVIRONMENT_IS_PTHREAD){var parentPort=worker_threads["parentPort"];parentPort.on("message",msg=>global.onmessage?.({data:msg}));Object.assign(globalThis,{self:global,postMessage:msg=>parentPort["postMessage"](msg)});process.on("uncaughtException",err=>{postMessage({cmd:"uncaughtException",error:err});process.exit(1)})}var startWorker;if(ENVIRONMENT_IS_PTHREAD){var initializedJS=false;self.onunhandledrejection=e=>{throw e.reason||e};async function handleMessage(e){try{var msgData=e["data"];var cmd=msgData.cmd;if(cmd==="load"){let messageQueue=[];self.onmessage=e=>messageQueue.push(e);startWorker=()=>{postMessage({cmd:"loaded"});for(let msg of messageQueue){handleMessage(msg)}self.onmessage=handleMessage};for(const handler of msgData.handlers){if(!Module[handler]||Module[handler].proxy){Module[handler]=(...args)=>{postMessage({cmd:"callHandler",handler,args})};if(handler=="print")out=Module[handler];if(handler=="printErr")err=Module[handler]}}wasmMemory=msgData.wasmMemory;updateMemoryViews();wasmModule=msgData.wasmModule;createWasm();run()}else if(cmd==="run"){establishStackSpace(msgData.pthread_ptr);__emscripten_thread_init(msgData.pthread_ptr,0,0,1,0,0);PThread.threadInitTLS();__emscripten_thread_mailbox_await(msgData.pthread_ptr);if(!initializedJS){initializedJS=true}try{await invokeEntryPoint(msgData.start_routine,msgData.arg)}catch(ex){if(ex!="unwind"){throw ex}}}else if(msgData.target==="setimmediate"){}else if(cmd==="checkMailbox"){if(initializedJS){checkMailbox()}}else if(cmd){err(`worker: received unknown command ${cmd}`);err(msgData)}}catch(ex){__emscripten_thread_crashed();throw ex}}self.onmessage=handleMessage}var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;var HEAP64,HEAPU64;var runtimeInitialized=false;function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function initMemory(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"]}else{var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||134217728;wasmMemory=new WebAssembly.Memory({initial:BigInt(INITIAL_MEMORY/65536),maximum:65536n,shared:true,address:"i64"})}updateMemoryViews()}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(ENVIRONMENT_IS_PTHREAD)return startWorker();if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function preMain(){}function postRun(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("wllama.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){assignWasmImports();if(!wasmImports.__instrumented){wasmImports.__instrumented=true;Asyncify.instrumentWasmImports(wasmImports)}var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);wasmExports=applySignatureConversions(wasmExports);registerTLSInit(wasmExports["_emscripten_tls_init"]);assignWasmExports(wasmExports);wasmModule=module;removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"],result["module"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst,mod))})})}if(ENVIRONMENT_IS_PTHREAD){var instance=new WebAssembly.Instance(wasmModule,getWasmImports());return receiveInstance(instance,wasmModule)}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var terminateWorker=worker=>{worker.terminate();worker.onmessage=e=>{}};var cleanupThread=pthread_ptr=>{var worker=PThread.pthreads[pthread_ptr];PThread.returnWorkerToPool(worker)};var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)};var spawnThread=threadParams=>{var worker=PThread.getNewWorker();if(!worker){return 6}PThread.runningWorkers.push(worker);PThread.pthreads[threadParams.pthread_ptr]=worker;worker.pthread_ptr=threadParams.pthread_ptr;var msg={cmd:"run",start_routine:threadParams.startRoutine,arg:threadParams.arg,pthread_ptr:threadParams.pthread_ptr};if(ENVIRONMENT_IS_NODE){worker.unref()}worker.postMessage(msg,threadParams.transferList);return 0};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var stackSave=()=>_emscripten_stack_get_current();var stackRestore=val=>__emscripten_stack_restore(val);var stackAlloc=sz=>__emscripten_stack_alloc(sz);var proxyToMainThread=(funcIndex,emAsmAddr,sync,...callArgs)=>{var serializedNumCallArgs=callArgs.length*2;var sp=stackSave();var args=stackAlloc(serializedNumCallArgs*8);var b=args/8;for(var i=0;i<callArgs.length;i++){var arg=callArgs[i];if(typeof arg=="bigint"){(growMemViews(),HEAP64)[b+2*i]=1n;(growMemViews(),HEAP64)[b+2*i+1]=arg}else{(growMemViews(),HEAP64)[b+2*i]=0n;(growMemViews(),HEAPF64)[b+2*i+1]=arg}}var rtn=__emscripten_run_js_on_main_thread(funcIndex,emAsmAddr,serializedNumCallArgs,args,sync);stackRestore(sp);return rtn};function _proc_exit(code){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(0,0,1,code);EXITSTATUS=code;if(!keepRuntimeAlive()){PThread.terminateAllThreads();Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))}function exitOnMainThread(returnCode){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(1,0,0,returnCode);_exit(returnCode)}var exitJS=(status,implicit)=>{EXITSTATUS=status;if(ENVIRONMENT_IS_PTHREAD){exitOnMainThread(status);throw"unwind"}_proc_exit(status)};var _exit=exitJS;var PThread={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init(){if(!ENVIRONMENT_IS_PTHREAD){PThread.initMainThread()}},initMainThread(){var pthreadPoolSize=Module["pthreadPoolSize"];while(pthreadPoolSize--){PThread.allocateUnusedWorker()}addOnPreRun(async()=>{var pthreadPoolReady=PThread.loadWasmModuleToAllWorkers();addRunDependency("loading-workers");await pthreadPoolReady;removeRunDependency("loading-workers")})},terminateAllThreads:()=>{for(var worker of PThread.runningWorkers){terminateWorker(worker)}for(var worker of PThread.unusedWorkers){terminateWorker(worker)}PThread.unusedWorkers=[];PThread.runningWorkers=[];PThread.pthreads={}},returnWorkerToPool:worker=>{var pthread_ptr=worker.pthread_ptr;delete PThread.pthreads[pthread_ptr];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);worker.pthread_ptr=0;__emscripten_thread_free_data(pthread_ptr)},threadInitTLS(){PThread.tlsInitFunctions.forEach(f=>f())},loadWasmModuleToWorker:worker=>new Promise(onFinishedLoading=>{worker.onmessage=e=>{var d=e["data"];var cmd=d.cmd;if(d.targetThread&&d.targetThread!=_pthread_self()){var targetWorker=PThread.pthreads[d.targetThread];if(targetWorker){targetWorker.postMessage(d,d.transferList)}else{err(`Internal error! Worker sent a message "${cmd}" to target pthread ${d.targetThread}, but that thread no longer exists!`)}return}if(cmd==="checkMailbox"){checkMailbox()}else if(cmd==="spawnThread"){spawnThread(d)}else if(cmd==="cleanupThread"){callUserCallback(()=>cleanupThread(d.thread))}else if(cmd==="loaded"){worker.loaded=true;if(ENVIRONMENT_IS_NODE&&!worker.pthread_ptr){worker.unref()}onFinishedLoading(worker)}else if(d.target==="setimmediate"){worker.postMessage(d)}else if(cmd==="uncaughtException"){worker.onerror(d.error)}else if(cmd==="callHandler"){Module[d.handler](...d.args)}else if(cmd){err(`worker sent an unknown command ${cmd}`)}};worker.onerror=e=>{var message="worker sent an error!";err(`${message} ${e.filename}:${e.lineno}: ${e.message}`);throw e};if(ENVIRONMENT_IS_NODE){worker.on("message",data=>worker.onmessage({data}));worker.on("error",e=>worker.onerror(e))}var handlers=[];var knownHandlers=["onExit","onAbort","print","printErr"];for(var handler of knownHandlers){if(Module.propertyIsEnumerable(handler)){handlers.push(handler)}}worker.postMessage({cmd:"load",handlers,wasmMemory,wasmModule})}),async loadWasmModuleToAllWorkers(){if(ENVIRONMENT_IS_PTHREAD){return}let pthreadPoolReady=Promise.all(PThread.unusedWorkers.map(PThread.loadWasmModuleToWorker));return pthreadPoolReady},allocateUnusedWorker(){var worker;var pthreadMainJs=_scriptName;if(Module["mainScriptUrlOrBlob"]){pthreadMainJs=Module["mainScriptUrlOrBlob"];if(typeof pthreadMainJs!="string"){pthreadMainJs=URL.createObjectURL(pthreadMainJs)}}worker=new Worker(pthreadMainJs,{workerData:"em-pthread",name:"em-pthread"});PThread.unusedWorkers.push(worker)},getNewWorker(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0])}return PThread.unusedWorkers.pop()}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);function establishStackSpace(pthread_ptr){var stackHigh=Number((growMemViews(),HEAPU64)[(pthread_ptr+88)/8]);var stackSize=Number((growMemViews(),HEAPU64)[(pthread_ptr+96)/8]);var stackLow=stackHigh-stackSize;_emscripten_stack_set_limits(stackHigh,stackLow);stackRestore(stackHigh)}var wasmTableMirror=[];var getWasmTableEntry=funcPtr=>{funcPtr=Number(funcPtr);var func=wasmTableMirror[funcPtr];if(!func){wasmTableMirror[funcPtr]=func=wasmTable.get(BigInt(funcPtr));if(Asyncify.isAsyncExport(func)){wasmTableMirror[funcPtr]=func=Asyncify.makeAsyncFunction(func)}}return func};var invokeEntryPoint=async(ptr,arg)=>{runtimeKeepaliveCounter=0;noExitRuntime=0;var result=(a1=>WebAssembly.promising(getWasmTableEntry(ptr)).call(null,BigInt(a1)))(arg);function finish(result){if(keepRuntimeAlive()){EXITSTATUS=result;return}__emscripten_thread_exit(result)}result=await result;finish(result)};invokeEntryPoint.isAsync=true;var noExitRuntime=true;var registerTLSInit=tlsInitFunc=>PThread.tlsInitFunctions.push(tlsInitFunc);var wasmMemory;function pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(2,0,1,pthread_ptr,attr,startRoutine,arg);return ___pthread_create_js(pthread_ptr,attr,startRoutine,arg)}var _emscripten_has_threading_support=()=>!!globalThis.SharedArrayBuffer;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function ___pthread_create_js(pthread_ptr,attr,startRoutine,arg){pthread_ptr=bigintToI53Checked(pthread_ptr);attr=bigintToI53Checked(attr);startRoutine=bigintToI53Checked(startRoutine);arg=bigintToI53Checked(arg);if(!_emscripten_has_threading_support()){return 6}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg)}if(error)return error;var threadParams={startRoutine,pthread_ptr,arg,transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);return 0}return spawnThread(threadParams)}var syscallGetVarargP=()=>{var ret=Number((growMemViews(),HEAPU64)[SYSCALLS.varargs/8]);SYSCALLS.varargs+=8;return ret};var syscallGetVarargI=()=>{var ret=(growMemViews(),HEAP32)[+SYSCALLS.varargs/4];SYSCALLS.varargs+=4;return ret};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\\/?|)([\\s\\S]*?)((?:\\.{1,2}|[^\\/]+?|)(\\.[^.\\/]*|))(?:[\\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\\/]+|\\/)\\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>view=>view.set(crypto.getRandomValues(new Uint8Array(view.byteLength)));var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;if(ignoreNul)return maxIdx;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead,ignoreNul);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.buffer instanceof ArrayBuffer?heapOrArray.subarray(idx,endPtr):heapOrArray.slice(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var zeroMemory=(ptr,size)=>(growMemViews(),HEAPU8).fill(0,ptr,ptr+size);var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var mmapAlloc=size=>{size=alignMemory(size,65536);var ptr=_emscripten_builtin_memalign(65536,size);if(ptr)zeroMemory(ptr,size);return ptr};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>"}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===(growMemViews(),HEAP8).buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===(growMemViews(),HEAP8).buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}(growMemViews(),HEAP8).set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var getUniqueRunDependency=id=>id;var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);addRunDependency(dep);try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url)}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}}finally{removeRunDependency(dep)}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror)};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done)}else{done(null)}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`)}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf)}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){data=new Uint8Array(intArrayFromString(data,true))}if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{abort("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url)}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,(growMemViews(),HEAP8),ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString((growMemViews(),HEAPU8),ptr,maxBytesToRead,ignoreNul):"";var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){(growMemViews(),HEAPU32)[buf/4]=stat.dev;(growMemViews(),HEAPU32)[(buf+4)/4]=stat.mode;(growMemViews(),HEAPU64)[(buf+8)/8]=BigInt(stat.nlink);(growMemViews(),HEAPU32)[(buf+16)/4]=stat.uid;(growMemViews(),HEAPU32)[(buf+20)/4]=stat.gid;(growMemViews(),HEAPU32)[(buf+24)/4]=stat.rdev;(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stat.size);(growMemViews(),HEAP32)[(buf+40)/4]=4096;(growMemViews(),HEAP32)[(buf+44)/4]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(Math.floor(atime/1e3));(growMemViews(),HEAPU64)[(buf+56)/8]=BigInt(atime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+64)/8]=BigInt(Math.floor(mtime/1e3));(growMemViews(),HEAPU64)[(buf+72)/8]=BigInt(mtime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+80)/8]=BigInt(Math.floor(ctime/1e3));(growMemViews(),HEAPU64)[(buf+88)/8]=BigInt(ctime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+96)/8]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){(growMemViews(),HEAPU32)[(buf+8)/4]=stats.bsize;(growMemViews(),HEAPU32)[(buf+72)/4]=stats.bsize;(growMemViews(),HEAP64)[(buf+16)/8]=BigInt(stats.blocks);(growMemViews(),HEAP64)[(buf+24)/8]=BigInt(stats.bfree);(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stats.bavail);(growMemViews(),HEAP64)[(buf+40)/8]=BigInt(stats.files);(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(stats.ffree);(growMemViews(),HEAPU32)[(buf+56)/4]=stats.fsid;(growMemViews(),HEAPU32)[(buf+80)/4]=stats.flags;(growMemViews(),HEAPU32)[(buf+64)/4]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=(growMemViews(),HEAPU8).slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(3,0,1,fd,cmd,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 5:{var arg=syscallGetVarargP();var offset=0;(growMemViews(),HEAP16)[(arg+offset)/2]=2;return 0}case 6:case 7:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_fstat64(fd,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(4,0,1,fd,buf);buf=bigintToI53Checked(buf);try{return SYSCALLS.writeStat(buf,FS.fstat(fd))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,(growMemViews(),HEAPU8),outPtr,maxBytesToWrite);function ___syscall_getcwd(buf,size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(5,0,1,buf,size);buf=bigintToI53Checked(buf);size=bigintToI53Checked(size);try{if(size===0)return-28;var cwd=FS.cwd();var cwdLengthInBytes=lengthBytesUTF8(cwd)+1;if(size<cwdLengthInBytes)return-68;stringToUTF8(cwd,buf,size);return cwdLengthInBytes}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_getdents64(fd,dirp,count){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(6,0,1,fd,dirp,count);dirp=bigintToI53Checked(dirp);count=bigintToI53Checked(count);try{var stream=SYSCALLS.getStreamFromFD(fd);stream.getdents||=FS.readdir(stream.path);var struct_size=280;var pos=0;var off=FS.llseek(stream,0,1);var startIdx=Math.floor(off/struct_size);var endIdx=Math.min(stream.getdents.length,startIdx+Math.floor(count/struct_size));for(var idx=startIdx;idx<endIdx;idx++){var id;var type;var name=stream.getdents[idx];if(name==="."){id=stream.node.id;type=4}else if(name===".."){var lookup=FS.lookupPath(stream.path,{parent:true});id=lookup.node.id;type=4}else{var child;try{child=FS.lookupNode(stream.node,name)}catch(e){if(e?.errno===28){continue}throw e}id=child.id;type=FS.isChrdev(child.mode)?2:FS.isDir(child.mode)?4:FS.isLink(child.mode)?10:8}(growMemViews(),HEAP64)[(dirp+pos)/8]=BigInt(id);(growMemViews(),HEAP64)[(dirp+pos+8)/8]=BigInt((idx+1)*struct_size);(growMemViews(),HEAP16)[(dirp+pos+16)/2]=280;(growMemViews(),HEAP8)[dirp+pos+18]=type;stringToUTF8(name,dirp+pos+19,256);pos+=struct_size}FS.llseek(stream,idx*struct_size,0);return pos}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(7,0,1,fd,op,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=termios.c_iflag||0;(growMemViews(),HEAP32)[(argp+4)/4]=termios.c_oflag||0;(growMemViews(),HEAP32)[(argp+8)/4]=termios.c_cflag||0;(growMemViews(),HEAP32)[(argp+12)/4]=termios.c_lflag||0;for(var i=0;i<32;i++){(growMemViews(),HEAP8)[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=(growMemViews(),HEAP32)[argp/4];var c_oflag=(growMemViews(),HEAP32)[(argp+4)/4];var c_cflag=(growMemViews(),HEAP32)[(argp+8)/4];var c_lflag=(growMemViews(),HEAP32)[(argp+12)/4];var c_cc=[];for(var i=0;i<32;i++){c_cc.push((growMemViews(),HEAP8)[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21537:case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();(growMemViews(),HEAP16)[argp/2]=winsize[0];(growMemViews(),HEAP16)[(argp+2)/2]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_lstat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(8,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.lstat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_newfstatat(dirfd,path,buf,flags){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(9,0,1,dirfd,path,buf,flags);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);var nofollow=flags&256;var allowEmpty=flags&4096;flags=flags&~6400;path=SYSCALLS.calculateAt(dirfd,path,allowEmpty);return SYSCALLS.writeStat(buf,nofollow?FS.lstat(path):FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(10,0,1,dirfd,path,flags,varargs);path=bigintToI53Checked(path);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_stat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(11,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __abort_js=()=>abort("");function __emscripten_init_main_thread_js(tb){tb=bigintToI53Checked(tb);__emscripten_thread_init(tb,!ENVIRONMENT_IS_WORKER,1,!ENVIRONMENT_IS_WEB,5242880,false);PThread.threadInitTLS()}var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};var maybeExit=()=>{if(!keepRuntimeAlive()){try{if(ENVIRONMENT_IS_PTHREAD){if(_pthread_self())__emscripten_thread_exit(EXITSTATUS);return}_exit(EXITSTATUS)}catch(e){handleException(e)}}};var callUserCallback=func=>{if(ABORT){return}try{func();maybeExit()}catch(e){handleException(e)}};function __emscripten_thread_mailbox_await(pthread_ptr){pthread_ptr=bigintToI53Checked(pthread_ptr);if(Atomics.waitAsync){var wait=Atomics.waitAsync((growMemViews(),HEAP32),pthread_ptr/4,pthread_ptr);wait.value.then(checkMailbox);var waitingAsync=pthread_ptr+228;Atomics.store((growMemViews(),HEAP32),waitingAsync/4,1)}}var checkMailbox=()=>callUserCallback(()=>{var pthread_ptr=_pthread_self();if(pthread_ptr){__emscripten_thread_mailbox_await(pthread_ptr);__emscripten_check_mailbox()}});function __emscripten_notify_mailbox_postmessage(targetThread,currThreadId){targetThread=bigintToI53Checked(targetThread);currThreadId=bigintToI53Checked(currThreadId);if(targetThread==currThreadId){setTimeout(checkMailbox)}else if(ENVIRONMENT_IS_PTHREAD){postMessage({targetThread,cmd:"checkMailbox"})}else{var worker=PThread.pthreads[targetThread];if(!worker){return}worker.postMessage({cmd:"checkMailbox"})}}var proxiedJSCallArgs=[];function __emscripten_receive_on_main_thread_js(funcIndex,emAsmAddr,callingThread,numCallArgs,args){emAsmAddr=bigintToI53Checked(emAsmAddr);callingThread=bigintToI53Checked(callingThread);args=bigintToI53Checked(args);numCallArgs/=2;proxiedJSCallArgs.length=numCallArgs;var b=args/8;for(var i=0;i<numCallArgs;i++){if((growMemViews(),HEAP64)[b+2*i]){proxiedJSCallArgs[i]=(growMemViews(),HEAP64)[b+2*i+1]}else{proxiedJSCallArgs[i]=(growMemViews(),HEAPF64)[b+2*i+1]}}var func=proxiedFunctionTable[funcIndex];PThread.currentProxiedOperationCallerThread=callingThread;var rtn=func(...proxiedJSCallArgs);PThread.currentProxiedOperationCallerThread=0;if(typeof rtn=="bigint"){rtn=bigintToI53Checked(rtn)}return rtn}function __emscripten_thread_cleanup(thread){thread=bigintToI53Checked(thread);if(!ENVIRONMENT_IS_PTHREAD)cleanupThread(thread);else postMessage({cmd:"cleanupThread",thread})}function __emscripten_thread_set_strongref(thread){thread=bigintToI53Checked(thread);if(ENVIRONMENT_IS_NODE){PThread.pthreads[thread].ref()}}var isLeapYear=year=>year%4===0&&(year%100!==0||year%400===0);var MONTH_DAYS_LEAP_CUMULATIVE=[0,31,60,91,121,152,182,213,244,274,305,335];var MONTH_DAYS_REGULAR_CUMULATIVE=[0,31,59,90,120,151,181,212,243,273,304,334];var ydayFromDate=date=>{var leap=isLeapYear(date.getFullYear());var monthDaysCumulative=leap?MONTH_DAYS_LEAP_CUMULATIVE:MONTH_DAYS_REGULAR_CUMULATIVE;var yday=monthDaysCumulative[date.getMonth()]+date.getDate()-1;return yday};function __localtime_js(time,tmPtr){time=bigintToI53Checked(time);tmPtr=bigintToI53Checked(tmPtr);var date=new Date(time*1e3);(growMemViews(),HEAP32)[tmPtr/4]=date.getSeconds();(growMemViews(),HEAP32)[(tmPtr+4)/4]=date.getMinutes();(growMemViews(),HEAP32)[(tmPtr+8)/4]=date.getHours();(growMemViews(),HEAP32)[(tmPtr+12)/4]=date.getDate();(growMemViews(),HEAP32)[(tmPtr+16)/4]=date.getMonth();(growMemViews(),HEAP32)[(tmPtr+20)/4]=date.getFullYear()-1900;(growMemViews(),HEAP32)[(tmPtr+24)/4]=date.getDay();var yday=ydayFromDate(date)|0;(growMemViews(),HEAP32)[(tmPtr+28)/4]=yday;(growMemViews(),HEAP64)[(tmPtr+40)/8]=BigInt(-(date.getTimezoneOffset()*60));var start=new Date(date.getFullYear(),0,1);var summerOffset=new Date(date.getFullYear(),6,1).getTimezoneOffset();var winterOffset=start.getTimezoneOffset();var dst=(summerOffset!=winterOffset&&date.getTimezoneOffset()==Math.min(winterOffset,summerOffset))|0;(growMemViews(),HEAP32)[(tmPtr+32)/4]=dst}function __mmap_js(len,prot,flags,fd,offset,allocated,addr){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(12,0,1,len,prot,flags,fd,offset,allocated,addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);allocated=bigintToI53Checked(allocated);addr=bigintToI53Checked(addr);try{var stream=SYSCALLS.getStreamFromFD(fd);var res=FS.mmap(stream,len,offset,prot,flags);var ptr=res.ptr;(growMemViews(),HEAP32)[allocated/4]=res.allocated;(growMemViews(),HEAPU64)[addr/8]=BigInt(ptr);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function __munmap_js(addr,len,prot,flags,fd,offset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(13,0,1,addr,len,prot,flags,fd,offset);addr=bigintToI53Checked(addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);try{var stream=SYSCALLS.getStreamFromFD(fd);if(prot&2){SYSCALLS.doMsync(addr,stream,len,flags,offset)}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __tzset_js=function(timezone,daylight,std_name,dst_name){timezone=bigintToI53Checked(timezone);daylight=bigintToI53Checked(daylight);std_name=bigintToI53Checked(std_name);dst_name=bigintToI53Checked(dst_name);var currentYear=(new Date).getFullYear();var winter=new Date(currentYear,0,1);var summer=new Date(currentYear,6,1);var winterOffset=winter.getTimezoneOffset();var summerOffset=summer.getTimezoneOffset();var stdTimezoneOffset=Math.max(winterOffset,summerOffset);(growMemViews(),HEAPU64)[timezone/8]=BigInt(stdTimezoneOffset*60);(growMemViews(),HEAP32)[daylight/4]=Number(winterOffset!=summerOffset);var extractZone=timezoneOffset=>{var sign=timezoneOffset>=0?"-":"+";var absOffset=Math.abs(timezoneOffset);var hours=String(Math.floor(absOffset/60)).padStart(2,"0");var minutes=String(absOffset%60).padStart(2,"0");return`UTC${sign}${hours}${minutes}`};var winterName=extractZone(winterOffset);var summerName=extractZone(summerOffset);if(summerOffset<winterOffset){stringToUTF8(winterName,std_name,17);stringToUTF8(summerName,dst_name,17)}else{stringToUTF8(winterName,dst_name,17);stringToUTF8(summerName,std_name,17)}};var _emscripten_get_now=()=>performance.timeOrigin+performance.now();var _emscripten_date_now=()=>Date.now();var nowIsMonotonic=1;var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;function _clock_time_get(clk_id,ignored_precision,ptime){ignored_precision=bigintToI53Checked(ignored_precision);ptime=bigintToI53Checked(ptime);if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now()}else if(nowIsMonotonic){now=_emscripten_get_now()}else{return 52}var nsec=Math.round(now*1e3*1e3);(growMemViews(),HEAP64)[ptime/8]=BigInt(nsec);return 0}var _emscripten_check_blocking_allowed=()=>{};var runtimeKeepalivePush=()=>{runtimeKeepaliveCounter+=1};var _emscripten_exit_with_live_runtime=()=>{runtimeKeepalivePush();throw"unwind"};var jsStackTrace=()=>(new Error).stack.toString();var getCallstack=flags=>{var callstack=jsStackTrace();var lines=callstack.split("\\n");callstack="";var firefoxRe=new RegExp("\\\\s*(.*?)@(.*?):([0-9]+):([0-9]+)");var chromeRe=new RegExp("\\\\s*at (.*?) \\\\((.*):(.*):(.*)\\\\)");for(var line of lines){var symbolName="";var file="";var lineno=0;var column=0;var parts=chromeRe.exec(line);if(parts?.length==5){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]}else{parts=firefoxRe.exec(line);if(parts?.length>=4){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]|0}else{callstack+=line+"\\n";continue}}if(symbolName=="_emscripten_log"||symbolName=="_emscripten_get_callstack"){callstack="";continue}if(flags&24){if(flags&64){file=file.substring(file.replace(/\\\\/g,"/").lastIndexOf("/")+1)}callstack+=`    at ${symbolName} (${file}:${lineno}:${column})\\n`}}callstack=callstack.replace(/\\s+$/,"");return callstack};function _emscripten_get_callstack(flags,str,maxbytes){str=bigintToI53Checked(str);var callstack=getCallstack(flags);if(!str||maxbytes<=0){return lengthBytesUTF8(callstack)+1}var bytesWrittenExcludingNull=stringToUTF8(callstack,str,maxbytes);return bytesWrittenExcludingNull+1}var getHeapMax=()=>4294967296;var _emscripten_get_heap_max=()=>BigInt(getHeapMax());var _emscripten_has_asyncify=()=>2;var _emscripten_num_logical_cores=()=>ENVIRONMENT_IS_NODE?require("os").cpus().length:navigator["hardwareConcurrency"];var growMemory=size=>{var oldHeapSize=wasmMemory.buffer.byteLength;var pages=(size-oldHeapSize+65535)/65536|0;try{wasmMemory.grow(BigInt(pages));updateMemoryViews();return 1}catch(e){}};function _emscripten_resize_heap(requestedSize){requestedSize=bigintToI53Checked(requestedSize);var oldSize=(growMemViews(),HEAPU8).length;if(requestedSize<=oldSize){return false}var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false}var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var writeI53ToI64=(ptr,num)=>{(growMemViews(),HEAPU32)[ptr/4]=num;var lower=(growMemViews(),HEAPU32)[ptr/4];(growMemViews(),HEAPU32)[(ptr+4)/4]=(num-lower)/4294967296};var stringToNewUTF8=str=>{var size=lengthBytesUTF8(str)+1;var ret=_malloc(size);if(ret)stringToUTF8(str,ret,size);return ret};var readI53FromI64=ptr=>(growMemViews(),HEAPU32)[ptr/4]+(growMemViews(),HEAP32)[(ptr+4)/4]*4294967296;var WebGPU={Internals:{jsObjects:[],jsObjectInsert:(ptr,jsObject)=>{WebGPU.Internals.jsObjects[ptr]=jsObject},bufferOnUnmaps:[],futures:[],futureInsert:(futureId,promise)=>{WebGPU.Internals.futures[futureId]=new Promise(resolve=>promise.finally(()=>resolve(futureId)))}},getJsObject:ptr=>{if(!ptr)return undefined;return WebGPU.Internals.jsObjects[ptr]},importJsAdapter:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateAdapter(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroup:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroup(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroupLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroupLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBuffer:(buffer,parentPtr=0)=>{assert(buffer.mapState==="unmapped");var bufferPtr=_emwgpuCreateBuffer(parentPtr);WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);return bufferPtr},importJsCommandBuffer:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandBuffer(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsCommandEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsDevice:(device,parentPtr=0)=>{var queuePtr=_emwgpuCreateQueue(parentPtr);var devicePtr=_emwgpuCreateDevice(parentPtr,queuePtr);WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);return devicePtr},importJsExternalTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateExternalTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsPipelineLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreatePipelineLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQuerySet:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQuerySet(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQueue:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQueue(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundle:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundle(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundleEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundleEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSampler:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSampler(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsShaderModule:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateShaderModule(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSurface:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSurface(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTextureView:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTextureView(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},errorCallback:(callback,type,message,userdata)=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(message);((a1,a2,a3)=>getWasmTableEntry(callback).call(null,a1,BigInt(a2),BigInt(a3)))(type,BigInt(messagePtr),userdata);stackRestore(sp)},iterateExtensions:(root,handlers)=>{for(var ptr=Number((growMemViews(),HEAPU64)[root/8]);ptr;ptr=Number((growMemViews(),HEAPU64)[ptr/8])){var sType=(growMemViews(),HEAP32)[(ptr+8)/4];var handler=handlers[sType](ptr)}},setStringView:(ptr,data,length)=>{(growMemViews(),HEAPU64)[ptr/8]=BigInt(data);(growMemViews(),HEAPU64)[(ptr+8)/8]=BigInt(length)},makeStringFromStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);return UTF8ToString(ptr,length)},makeStringFromOptionalStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);if(!ptr){if(length===0){return""}return undefined}return UTF8ToString(ptr,length)},makeColor:ptr=>({r:(growMemViews(),HEAPF64)[ptr/8],g:(growMemViews(),HEAPF64)[(ptr+8)/8],b:(growMemViews(),HEAPF64)[(ptr+16)/8],a:(growMemViews(),HEAPF64)[(ptr+24)/8]}),makeExtent3D:ptr=>({width:(growMemViews(),HEAPU32)[ptr/4],height:(growMemViews(),HEAPU32)[(ptr+4)/4],depthOrArrayLayers:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeOrigin3D:ptr=>({x:(growMemViews(),HEAPU32)[ptr/4],y:(growMemViews(),HEAPU32)[(ptr+4)/4],z:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeTexelCopyTextureInfo:ptr=>({texture:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[ptr/8])),mipLevel:(growMemViews(),HEAPU32)[(ptr+8)/4],origin:WebGPU.makeOrigin3D(ptr+12),aspect:WebGPU.TextureAspect[(growMemViews(),HEAP32)[(ptr+24)/4]]}),makeTexelCopyBufferLayout:ptr=>{var bytesPerRow=(growMemViews(),HEAPU32)[(ptr+8)/4];var rowsPerImage=(growMemViews(),HEAPU32)[(ptr+12)/4];return{offset:readI53FromI64(ptr),bytesPerRow:bytesPerRow===4294967295?undefined:bytesPerRow,rowsPerImage:rowsPerImage===4294967295?undefined:rowsPerImage}},makeTexelCopyBufferInfo:ptr=>{var layoutPtr=ptr+0;var bufferCopyView=WebGPU.makeTexelCopyBufferLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+16)/8]));return bufferCopyView},makePassTimestampWrites:ptr=>{if(ptr===0)return undefined;return{querySet:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),beginningOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+16)/4],endOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+20)/4]}},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+32*i;var key=WebGPU.makeStringFromStringView(entryPtr+8);constants[key]=(growMemViews(),HEAPF64)[(entryPtr+24)/8]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.getJsObject(layoutPtr)},makeComputeState:ptr=>{if(!ptr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(ptr+32)/8]),Number((growMemViews(),HEAPU64)[(ptr+40)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(ptr+16)};return desc},makeComputePipelineDesc:descriptor=>{var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),compute:WebGPU.makeComputeState(descriptor+32)};return desc},makeRenderPipelineDesc:descriptor=>{function makePrimitiveState(psPtr){if(!psPtr)return undefined;return{topology:WebGPU.PrimitiveTopology[(growMemViews(),HEAP32)[(psPtr+8)/4]],stripIndexFormat:WebGPU.IndexFormat[(growMemViews(),HEAP32)[(psPtr+12)/4]],frontFace:WebGPU.FrontFace[(growMemViews(),HEAP32)[(psPtr+16)/4]],cullMode:WebGPU.CullMode[(growMemViews(),HEAP32)[(psPtr+20)/4]],unclippedDepth:!!(growMemViews(),HEAPU32)[(psPtr+24)/4]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{operation:WebGPU.BlendOperation[(growMemViews(),HEAP32)[bdPtr/4]],srcFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+4)/4]],dstFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+8)/4]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{alpha:makeBlendComponent(bsPtr+12),color:makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var format=WebGPU.TextureFormat[(growMemViews(),HEAP32)[(csPtr+8)/4]];return format?{format,blend:makeBlendState(Number((growMemViews(),HEAPU64)[(csPtr+16)/8])),writeMask:(growMemViews(),HEAPU32)[(csPtr+24)/4]}:undefined}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+32*i))}return states}function makeStencilStateFace(ssfPtr){return{compare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[ssfPtr/4]],failOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+4)/4]],depthFailOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+8)/4]],passOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+12)/4]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{format:WebGPU.TextureFormat[(growMemViews(),HEAP32)[(dssPtr+8)/4]],depthWriteEnabled:!!(growMemViews(),HEAPU32)[(dssPtr+12)/4],depthCompare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[(dssPtr+16)/4]],stencilFront:makeStencilStateFace(dssPtr+20),stencilBack:makeStencilStateFace(dssPtr+36),stencilReadMask:(growMemViews(),HEAPU32)[(dssPtr+52)/4],stencilWriteMask:(growMemViews(),HEAPU32)[(dssPtr+56)/4],depthBias:(growMemViews(),HEAP32)[(dssPtr+60)/4],depthBiasSlopeScale:(growMemViews(),HEAPF32)[(dssPtr+64)/4],depthBiasClamp:(growMemViews(),HEAPF32)[(dssPtr+68)/4]}}function makeVertexAttribute(vaPtr){return{format:WebGPU.VertexFormat[(growMemViews(),HEAP32)[(vaPtr+8)/4]],offset:readI53FromI64(vaPtr+16),shaderLocation:(growMemViews(),HEAPU32)[(vaPtr+24)/4]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*32))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepMode=WebGPU.VertexStepMode[(growMemViews(),HEAP32)[(vbPtr+8)/4]];var attributeCount=Number((growMemViews(),HEAPU64)[(vbPtr+24)/8]);if(!stepMode&&!attributeCount){return null}return{arrayStride:readI53FromI64(vbPtr+16),stepMode,attributes:makeVertexAttributes(attributeCount,Number((growMemViews(),HEAPU64)[(vbPtr+32)/8]))}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*40))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(viPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(viPtr+32)/8]),Number((growMemViews(),HEAPU64)[(viPtr+40)/8])),buffers:makeVertexBuffers(Number((growMemViews(),HEAPU64)[(viPtr+48)/8]),Number((growMemViews(),HEAPU64)[(viPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(viPtr+16)};return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{count:(growMemViews(),HEAPU32)[(msPtr+8)/4],mask:(growMemViews(),HEAPU32)[(msPtr+12)/4],alphaToCoverageEnabled:!!(growMemViews(),HEAPU32)[(msPtr+16)/4]}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(fsPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(fsPtr+32)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+40)/8])),targets:makeColorStates(Number((growMemViews(),HEAPU64)[(fsPtr+48)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(fsPtr+16)};return desc}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),vertex:makeVertexState(descriptor+32),primitive:makePrimitiveState(descriptor+96),depthStencil:makeDepthStencilState(Number((growMemViews(),HEAPU64)[(descriptor+128)/8])),multisample:makeMultisampleState(descriptor+136),fragment:makeFragmentState(Number((growMemViews(),HEAPU64)[(descriptor+160)/8]))};return desc},fillLimitStruct:(limits,limitsOutPtr)=>{var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsOutPtr/8]);function setLimitValueU32(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;(growMemViews(),HEAPU32)[(basePtr+limitOffset)/4]=limitValue}function setLimitValueU64(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;writeI53ToI64(basePtr+limitOffset,limitValue)}setLimitValueU32("maxTextureDimension1D",limitsOutPtr,8);setLimitValueU32("maxTextureDimension2D",limitsOutPtr,12);setLimitValueU32("maxTextureDimension3D",limitsOutPtr,16);setLimitValueU32("maxTextureArrayLayers",limitsOutPtr,20);setLimitValueU32("maxBindGroups",limitsOutPtr,24);setLimitValueU32("maxBindGroupsPlusVertexBuffers",limitsOutPtr,28);setLimitValueU32("maxBindingsPerBindGroup",limitsOutPtr,32);setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout",limitsOutPtr,36);setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout",limitsOutPtr,40);setLimitValueU32("maxSampledTexturesPerShaderStage",limitsOutPtr,44);setLimitValueU32("maxSamplersPerShaderStage",limitsOutPtr,48);setLimitValueU32("maxStorageBuffersPerShaderStage",limitsOutPtr,52);setLimitValueU32("maxStorageTexturesPerShaderStage",limitsOutPtr,56);setLimitValueU32("maxUniformBuffersPerShaderStage",limitsOutPtr,60);setLimitValueU32("minUniformBufferOffsetAlignment",limitsOutPtr,80);setLimitValueU32("minStorageBufferOffsetAlignment",limitsOutPtr,84);setLimitValueU64("maxUniformBufferBindingSize",limitsOutPtr,64);setLimitValueU64("maxStorageBufferBindingSize",limitsOutPtr,72);setLimitValueU32("maxVertexBuffers",limitsOutPtr,88);setLimitValueU64("maxBufferSize",limitsOutPtr,96);setLimitValueU32("maxVertexAttributes",limitsOutPtr,104);setLimitValueU32("maxVertexBufferArrayStride",limitsOutPtr,108);setLimitValueU32("maxInterStageShaderVariables",limitsOutPtr,112);setLimitValueU32("maxColorAttachments",limitsOutPtr,116);setLimitValueU32("maxColorAttachmentBytesPerSample",limitsOutPtr,120);setLimitValueU32("maxComputeWorkgroupStorageSize",limitsOutPtr,124);setLimitValueU32("maxComputeInvocationsPerWorkgroup",limitsOutPtr,128);setLimitValueU32("maxComputeWorkgroupSizeX",limitsOutPtr,132);setLimitValueU32("maxComputeWorkgroupSizeY",limitsOutPtr,136);setLimitValueU32("maxComputeWorkgroupSizeZ",limitsOutPtr,140);setLimitValueU32("maxComputeWorkgroupsPerDimension",limitsOutPtr,144);setLimitValueU32("maxImmediateSize",limitsOutPtr,148);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;setLimitValueU32("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20,limits.maxStorageTexturesPerShaderStage);setLimitValueU32("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28,limits.maxStorageTexturesPerShaderStage)}},fillAdapterInfoStruct:(info,infoStruct)=>{(growMemViews(),HEAPU32)[(infoStruct+88)/4]=info.subgroupMinSize;(growMemViews(),HEAPU32)[(infoStruct+92)/4]=info.subgroupMaxSize;var strs=info.vendor+info.architecture+info.device+info.description;var strPtr=stringToNewUTF8(strs);var vendorLen=lengthBytesUTF8(info.vendor);WebGPU.setStringView(infoStruct+8,strPtr,vendorLen);strPtr+=vendorLen;var architectureLen=lengthBytesUTF8(info.architecture);WebGPU.setStringView(infoStruct+24,strPtr,architectureLen);strPtr+=architectureLen;var deviceLen=lengthBytesUTF8(info.device);WebGPU.setStringView(infoStruct+40,strPtr,deviceLen);strPtr+=deviceLen;var descriptionLen=lengthBytesUTF8(info.description);WebGPU.setStringView(infoStruct+56,strPtr,descriptionLen);strPtr+=descriptionLen;(growMemViews(),HEAP32)[(infoStruct+72)/4]=2;var adapterType=info.isFallbackAdapter?3:4;(growMemViews(),HEAP32)[(infoStruct+76)/4]=adapterType;(growMemViews(),HEAPU32)[(infoStruct+80)/4]=0;(growMemViews(),HEAPU32)[(infoStruct+84)/4]=0},AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1-alpha","one-minus-src1-alpha"],BlendOperation:[,"add","subtract","reverse-subtract","min","max"],BufferBindingType:[,,"uniform","storage","read-only-storage"],BufferMapState:[,"unmapped","pending","mapped"],CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],CompilationInfoRequestStatus:[,"success","callback-cancelled"],ComponentSwizzle:[,"0","1","r","g","b","a"],CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],CullMode:[,"none","front","back"],ErrorFilter:[,"validation","out-of-memory","internal"],FeatureLevel:[,"compatibility","core"],FeatureName:{1:"core-features-and-limits",2:"depth-clip-control",3:"depth32float-stencil8",4:"texture-compression-bc",5:"texture-compression-bc-sliced-3d",6:"texture-compression-etc2",7:"texture-compression-astc",8:"texture-compression-astc-sliced-3d",9:"timestamp-query",10:"indirect-first-instance",11:"shader-f16",12:"rg11b10ufloat-renderable",13:"bgra8unorm-storage",14:"float32-filterable",15:"float32-blendable",16:"clip-distances",17:"dual-source-blending",18:"subgroups",19:"texture-formats-tier1",20:"texture-formats-tier2",21:"primitive-index",22:"texture-component-swizzle",327692:"chromium-experimental-unorm16-texture-formats",327729:"chromium-experimental-multi-draw-indirect"},FilterMode:[,"nearest","linear"],FrontFace:[,"ccw","cw"],IndexFormat:[,"uint16","uint32"],InstanceFeatureName:[,"timed-wait-any","shader-source-spirv","multiple-devices-per-adapter"],LoadOp:[,"load","clear"],MipmapFilterMode:[,"nearest","linear"],OptionalBool:["false","true"],PowerPreference:[,"low-power","high-performance"],PredefinedColorSpace:[,"srgb","display-p3"],PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:[,"occlusion","timestamp"],SamplerBindingType:[,,"filtering","non-filtering","comparison"],Status:[,"success","error"],StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,,"write-only","read-only","read-write"],StoreOp:[,"store","discard"],SurfaceGetCurrentTextureStatus:[,"success-optimal","success-suboptimal","timeout","outdated","lost","error"],TextureAspect:[,"all","stencil-only","depth-only"],TextureDimension:[,"1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16unorm","r16snorm","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16unorm","rg16snorm","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16unorm","rgba16snorm","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],ToneMappingMode:[,"standard","extended"],VertexFormat:[,"uint8","uint8x2","uint8x4","sint8","sint8x2","sint8x4","unorm8","unorm8x2","unorm8x4","snorm8","snorm8x2","snorm8x4","uint16","uint16x2","uint16x4","sint16","sint16x2","sint16x4","unorm16","unorm16x2","unorm16x4","snorm16","snorm16x2","snorm16x4","float16","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4","unorm10-10-10-2","unorm8x4-bgra"],VertexStepMode:[,"vertex","instance"],WGSLLanguageFeatureName:[,"readonly_and_readwrite_storage_textures","packed_4x8_integer_dot_product","unrestricted_pointer_parameters","pointer_composite_access","uniform_buffer_standard_layout","subgroup_id","texture_and_sampler_let","subgroup_uniformity","texture_formats_tier1"]};var emwgpuStringToInt_DeviceLostReason={undefined:1,unknown:1,destroyed:2};var runtimeKeepalivePop=()=>{runtimeKeepaliveCounter-=1};function _emwgpuAdapterRequestDevice(adapterPtr,futureId,deviceLostFutureId,devicePtr,queuePtr,descriptor){adapterPtr=bigintToI53Checked(adapterPtr);futureId=bigintToI53Checked(futureId);deviceLostFutureId=bigintToI53Checked(deviceLostFutureId);devicePtr=bigintToI53Checked(devicePtr);queuePtr=bigintToI53Checked(queuePtr);descriptor=bigintToI53Checked(descriptor);var adapter=WebGPU.getJsObject(adapterPtr);var desc={};if(descriptor){var requiredFeatureCount=Number((growMemViews(),HEAPU64)[(descriptor+24)/8]);if(requiredFeatureCount){var requiredFeaturesPtr=Number((growMemViews(),HEAPU64)[(descriptor+32)/8]);desc["requiredFeatures"]=Array.from((growMemViews(),HEAPU32).subarray(requiredFeaturesPtr/4,(requiredFeaturesPtr+requiredFeatureCount*4)/4),feature=>WebGPU.FeatureName[feature])}var limitsPtr=Number((growMemViews(),HEAPU64)[(descriptor+40)/8]);if(limitsPtr){var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsPtr/8]);var requiredLimits={};function setLimitU32IfDefined(name,basePtr,limitOffset,ignoreIfZero=false){var ptr=basePtr+limitOffset;var value=(growMemViews(),HEAPU32)[ptr/4];if(value!=4294967295&&(!ignoreIfZero||value!=0)){requiredLimits[name]=value}}function setLimitU64IfDefined(name,basePtr,limitOffset){var ptr=basePtr+limitOffset;var limitPart1=(growMemViews(),HEAPU32)[ptr/4];var limitPart2=(growMemViews(),HEAPU32)[(ptr+4)/4];if(limitPart1!=4294967295||limitPart2!=4294967295){requiredLimits[name]=readI53FromI64(ptr)}}setLimitU32IfDefined("maxTextureDimension1D",limitsPtr,8);setLimitU32IfDefined("maxTextureDimension2D",limitsPtr,12);setLimitU32IfDefined("maxTextureDimension3D",limitsPtr,16);setLimitU32IfDefined("maxTextureArrayLayers",limitsPtr,20);setLimitU32IfDefined("maxBindGroups",limitsPtr,24);setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers",limitsPtr,28);setLimitU32IfDefined("maxBindingsPerBindGroup",limitsPtr,32);setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout",limitsPtr,36);setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout",limitsPtr,40);setLimitU32IfDefined("maxSampledTexturesPerShaderStage",limitsPtr,44);setLimitU32IfDefined("maxSamplersPerShaderStage",limitsPtr,48);setLimitU32IfDefined("maxStorageBuffersPerShaderStage",limitsPtr,52);setLimitU32IfDefined("maxStorageTexturesPerShaderStage",limitsPtr,56);setLimitU32IfDefined("maxUniformBuffersPerShaderStage",limitsPtr,60);setLimitU32IfDefined("minUniformBufferOffsetAlignment",limitsPtr,80);setLimitU32IfDefined("minStorageBufferOffsetAlignment",limitsPtr,84);setLimitU64IfDefined("maxUniformBufferBindingSize",limitsPtr,64);setLimitU64IfDefined("maxStorageBufferBindingSize",limitsPtr,72);setLimitU32IfDefined("maxVertexBuffers",limitsPtr,88);setLimitU64IfDefined("maxBufferSize",limitsPtr,96);setLimitU32IfDefined("maxVertexAttributes",limitsPtr,104);setLimitU32IfDefined("maxVertexBufferArrayStride",limitsPtr,108);setLimitU32IfDefined("maxInterStageShaderVariables",limitsPtr,112);setLimitU32IfDefined("maxColorAttachments",limitsPtr,116);setLimitU32IfDefined("maxColorAttachmentBytesPerSample",limitsPtr,120);setLimitU32IfDefined("maxComputeWorkgroupStorageSize",limitsPtr,124);setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup",limitsPtr,128);setLimitU32IfDefined("maxComputeWorkgroupSizeX",limitsPtr,132);setLimitU32IfDefined("maxComputeWorkgroupSizeY",limitsPtr,136);setLimitU32IfDefined("maxComputeWorkgroupSizeZ",limitsPtr,140);setLimitU32IfDefined("maxComputeWorkgroupsPerDimension",limitsPtr,144);setLimitU32IfDefined("maxImmediateSize",limitsPtr,148,true);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;if("maxStorageBuffersInVertexStage"in GPUSupportedLimits.prototype){setLimitU32IfDefined("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16);setLimitU32IfDefined("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20);setLimitU32IfDefined("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24);setLimitU32IfDefined("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28)}}desc["requiredLimits"]=requiredLimits}var defaultQueuePtr=Number((growMemViews(),HEAPU64)[(descriptor+48)/8]);if(defaultQueuePtr){var defaultQueueDesc={label:WebGPU.makeStringFromOptionalStringView(defaultQueuePtr+8)};desc["defaultQueue"]=defaultQueueDesc}desc["label"]=WebGPU.makeStringFromOptionalStringView(descriptor+8)}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,adapter.requestDevice(desc).then(device=>{runtimeKeepalivePop();callUserCallback(()=>{WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);devicePtr=BigInt(devicePtr);WebGPU.Internals.futureInsert(deviceLostFutureId,device.lost.then(info=>{callUserCallback(()=>{device.onuncapturederror=ev=>{};var sp=stackSave();var messagePtr=stringToUTF8OnStack(info.message);_emwgpuOnDeviceLostCompleted(deviceLostFutureId,emwgpuStringToInt_DeviceLostReason[info.reason],BigInt(messagePtr));stackRestore(sp)})}));device.onuncapturederror=ev=>{var type=5;if(ev.error instanceof GPUValidationError)type=2;else if(ev.error instanceof GPUOutOfMemoryError)type=3;else if(ev.error instanceof GPUInternalError)type=4;var sp=stackSave();var messagePtr=stringToUTF8OnStack(ev.error.message);_emwgpuOnUncapturedError(BigInt(devicePtr),type,BigInt(messagePtr));stackRestore(sp)};_emwgpuOnRequestDeviceCompleted(futureId,1,BigInt(devicePtr),0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestDeviceCompleted(futureId,3,BigInt(devicePtr),BigInt(messagePtr));if(deviceLostFutureId){_emwgpuOnDeviceLostCompleted(deviceLostFutureId,4,BigInt(messagePtr))}stackRestore(sp)})}))}function _emwgpuBufferDestroy(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(onUnmap){for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]}buffer.destroy()}var warnOnce=text=>{warnOnce.shown||={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;if(ENVIRONMENT_IS_NODE)text="warning: "+text;err(text)}};var _emwgpuBufferGetConstMappedRange=function(bufferPtr,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);if(size==-1)size=undefined;var mapped;try{mapped=buffer.getMappedRange(offset,size)}catch(ex){return 0n}var data=_memalign(16,mapped.byteLength);(growMemViews(),HEAPU8).set(new Uint8Array(mapped),data);WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(()=>_free(data));return data})();return BigInt(ret)};var _emwgpuBufferMapAsync=function(bufferPtr,futureId,mode,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);futureId=bigintToI53Checked(futureId);mode=bigintToI53Checked(mode);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var buffer=WebGPU.getJsObject(bufferPtr);WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[];if(size==-1)size=undefined;runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,buffer.mapAsync(mode,offset,size).then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnMapAsyncCompleted(futureId,1,0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);var status=ex.name==="AbortError"?4:ex.name==="OperationError"?3:0;_emwgpuOnMapAsyncCompleted(futureId,status,BigInt(messagePtr));delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]})}))};function _emwgpuBufferUnmap(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(!onUnmap){return}for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];buffer.unmap()}function _emwgpuDelete(ptr){ptr=bigintToI53Checked(ptr);delete WebGPU.Internals.jsObjects[ptr]}function _emwgpuDeviceCreateBuffer(devicePtr,descriptor,bufferPtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);bufferPtr=bigintToI53Checked(bufferPtr);var mappedAtCreation=!!(growMemViews(),HEAPU32)[(descriptor+40)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),usage:(growMemViews(),HEAPU32)[(descriptor+24)/4],size:readI53FromI64(descriptor+32),mappedAtCreation};var device=WebGPU.getJsObject(devicePtr);var buffer;try{buffer=device.createBuffer(desc)}catch(ex){return false}WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);if(mappedAtCreation){WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[]}return true}function _emwgpuDeviceCreateShaderModule(devicePtr,descriptor,shaderModulePtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);shaderModulePtr=bigintToI53Checked(shaderModulePtr);var nextInChainPtr=Number((growMemViews(),HEAPU64)[descriptor/8]);var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),code:""};switch(sType){case 2:{desc["code"]=WebGPU.makeStringFromStringView(nextInChainPtr+16);break}}var device=WebGPU.getJsObject(devicePtr);WebGPU.Internals.jsObjectInsert(shaderModulePtr,device.createShaderModule(desc))}var _emwgpuDeviceDestroy=devicePtr=>{const device=WebGPU.getJsObject(devicePtr);device.onuncapturederror=null;device.destroy()};function _emwgpuInstanceRequestAdapter(instancePtr,futureId,options,adapterPtr){instancePtr=bigintToI53Checked(instancePtr);futureId=bigintToI53Checked(futureId);options=bigintToI53Checked(options);adapterPtr=bigintToI53Checked(adapterPtr);var opts;if(options){opts={featureLevel:WebGPU.FeatureLevel[(growMemViews(),HEAP32)[(options+8)/4]],powerPreference:WebGPU.PowerPreference[(growMemViews(),HEAP32)[(options+12)/4]],forceFallbackAdapter:!!(growMemViews(),HEAPU32)[(options+16)/4]};var nextInChainPtr=Number((growMemViews(),HEAPU64)[options/8]);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var webxrOptions=nextInChainPtr;opts.xrCompatible=!!(growMemViews(),HEAPU32)[(webxrOptions+16)/4]}}if(!("gpu"in navigator)){var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp);return}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,navigator.gpu.requestAdapter(opts).then(adapter=>{runtimeKeepalivePop();callUserCallback(()=>{if(adapter){WebGPU.Internals.jsObjectInsert(adapterPtr,adapter);_emwgpuOnRequestAdapterCompleted(futureId,1,BigInt(adapterPtr),0n)}else{var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)}})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestAdapterCompleted(futureId,4,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)})}))}var _emwgpuQueueOnSubmittedWorkDone=function(queuePtr,futureId){queuePtr=bigintToI53Checked(queuePtr);futureId=bigintToI53Checked(futureId);var queue=WebGPU.getJsObject(queuePtr);runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,queue.onSubmittedWorkDone().then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnWorkDoneCompleted(futureId,1)})}))};var _emwgpuWaitAny=function(futurePtr,futureCount,timeoutMSPtr){futurePtr=bigintToI53Checked(futurePtr);futureCount=bigintToI53Checked(futureCount);timeoutMSPtr=bigintToI53Checked(timeoutMSPtr);return Asyncify.handleAsync(async()=>{var promises=[];if(timeoutMSPtr){var timeoutMS=(growMemViews(),HEAP32)[timeoutMSPtr/4];promises.length=futureCount+1;promises[futureCount]=new Promise(resolve=>setTimeout(resolve,timeoutMS,0))}else{promises.length=futureCount}for(var i=0;i<futureCount;++i){var futureId=readI53FromI64(futurePtr+i*8);if(!(futureId in WebGPU.Internals.futures)){return futureId}promises[i]=WebGPU.Internals.futures[futureId]}const firstResolvedFuture=await Promise.race(promises);delete WebGPU.Internals.futures[firstResolvedFuture];return firstResolvedFuture})};_emwgpuWaitAny.isAsync=true;var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x]}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`)}getEnvStrings.strings=strings}return getEnvStrings.strings};function _environ_get(__environ,environ_buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(14,0,1,__environ,environ_buf);__environ=bigintToI53Checked(__environ);environ_buf=bigintToI53Checked(environ_buf);var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;(growMemViews(),HEAPU64)[(__environ+envp)/8]=BigInt(ptr);bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=8}return 0}function _environ_sizes_get(penviron_count,penviron_buf_size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(15,0,1,penviron_count,penviron_buf_size);penviron_count=bigintToI53Checked(penviron_count);penviron_buf_size=bigintToI53Checked(penviron_buf_size);var strings=getEnvStrings();(growMemViews(),HEAPU64)[penviron_count/8]=BigInt(strings.length);var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1}(growMemViews(),HEAPU64)[penviron_buf_size/8]=BigInt(bufSize);return 0}function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(16,0,1,fd);try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.read(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(17,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(18,0,1,fd,offset,whence,newOffset);offset=bigintToI53Checked(offset);newOffset=bigintToI53Checked(newOffset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);(growMemViews(),HEAP64)[newOffset/8]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.write(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(19,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _random_get(buffer,size){buffer=bigintToI53Checked(buffer);size=bigintToI53Checked(size);try{randomFill((growMemViews(),HEAPU8).subarray(buffer,buffer+size));return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _wgpuAdapterGetInfo(adapterPtr,info){adapterPtr=bigintToI53Checked(adapterPtr);info=bigintToI53Checked(info);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillAdapterInfoStruct(adapter.info,info);return 1}function _wgpuAdapterGetLimits(adapterPtr,limitsOutPtr){adapterPtr=bigintToI53Checked(adapterPtr);limitsOutPtr=bigintToI53Checked(limitsOutPtr);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillLimitStruct(adapter.limits,limitsOutPtr);return 1}function _wgpuAdapterHasFeature(adapterPtr,featureEnumValue){adapterPtr=bigintToI53Checked(adapterPtr);var adapter=WebGPU.getJsObject(adapterPtr);return adapter.features.has(WebGPU.FeatureName[featureEnumValue])}var _wgpuBufferGetSize=function(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);return buffer.size})();return BigInt(ret)};var _wgpuCommandEncoderBeginComputePass=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),timestampWrites:WebGPU.makePassTimestampWrites(Number((growMemViews(),HEAPU64)[(descriptor+24)/8]))}}var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateComputePassEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.beginComputePass(desc));return ptr})();return BigInt(ret)};function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr,srcPtr,srcOffset,dstPtr,dstOffset,size){encoderPtr=bigintToI53Checked(encoderPtr);srcPtr=bigintToI53Checked(srcPtr);srcOffset=bigintToI53Checked(srcOffset);dstPtr=bigintToI53Checked(dstPtr);dstOffset=bigintToI53Checked(dstOffset);size=bigintToI53Checked(size);var commandEncoder=WebGPU.getJsObject(encoderPtr);var src=WebGPU.getJsObject(srcPtr);var dst=WebGPU.getJsObject(dstPtr);commandEncoder.copyBufferToBuffer(src,srcOffset,dst,dstOffset,size)}var _wgpuCommandEncoderFinish=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateCommandBuffer(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.finish());return ptr})();return BigInt(ret)};function _wgpuComputePassEncoderDispatchWorkgroups(passPtr,x,y,z){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.dispatchWorkgroups(x,y,z)}function _wgpuComputePassEncoderEnd(passPtr){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.end()}function _wgpuComputePassEncoderSetBindGroup(passPtr,groupIndex,groupPtr,dynamicOffsetCount,dynamicOffsetsPtr){passPtr=bigintToI53Checked(passPtr);groupPtr=bigintToI53Checked(groupPtr);dynamicOffsetCount=bigintToI53Checked(dynamicOffsetCount);dynamicOffsetsPtr=bigintToI53Checked(dynamicOffsetsPtr);var pass=WebGPU.getJsObject(passPtr);var group=WebGPU.getJsObject(groupPtr);if(dynamicOffsetCount==0){pass.setBindGroup(groupIndex,group)}else{pass.setBindGroup(groupIndex,group,(growMemViews(),HEAPU32),dynamicOffsetsPtr/4,dynamicOffsetCount)}}function _wgpuComputePassEncoderSetPipeline(passPtr,pipelinePtr){passPtr=bigintToI53Checked(passPtr);pipelinePtr=bigintToI53Checked(pipelinePtr);var pass=WebGPU.getJsObject(passPtr);var pipeline=WebGPU.getJsObject(pipelinePtr);pass.setPipeline(pipeline)}var _wgpuComputePipelineGetBindGroupLayout=function(pipelinePtr,groupIndex){pipelinePtr=bigintToI53Checked(pipelinePtr);var ret=(()=>{var pipeline=WebGPU.getJsObject(pipelinePtr);var ptr=_emwgpuCreateBindGroupLayout(0n);WebGPU.Internals.jsObjectInsert(ptr,pipeline.getBindGroupLayout(groupIndex));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateBindGroup=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{function makeEntry(entryPtr){var bufferPtr=Number((growMemViews(),HEAPU64)[(entryPtr+16)/8]);var samplerPtr=Number((growMemViews(),HEAPU64)[(entryPtr+40)/8]);var textureViewPtr=Number((growMemViews(),HEAPU64)[(entryPtr+48)/8]);var externalTexturePtr=0;WebGPU.iterateExtensions(entryPtr,{327681:ptr=>{externalTexturePtr=Number((growMemViews(),HEAPU64)[(ptr+16)/8])}});var resource;if(bufferPtr){var size=readI53FromI64(entryPtr+32);if(size==-1)size=undefined;resource={buffer:WebGPU.getJsObject(bufferPtr),offset:readI53FromI64(entryPtr+24),size}}else{resource=WebGPU.getJsObject(samplerPtr||textureViewPtr||externalTexturePtr)}return{binding:(growMemViews(),HEAPU32)[(entryPtr+8)/4],resource}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+56*i))}return entries}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),entries:makeEntries(Number((growMemViews(),HEAPU64)[(descriptor+32)/8]),Number((growMemViews(),HEAPU64)[(descriptor+40)/8]))};var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateBindGroup(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createBindGroup(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateCommandEncoder=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8)}}var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateCommandEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createCommandEncoder(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateComputePipeline=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc=WebGPU.makeComputePipelineDesc(descriptor);var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateComputePipeline(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createComputePipeline(desc));return ptr})();return BigInt(ret)};function _wgpuInstanceHasWGSLLanguageFeature(instance,featureEnumValue){instance=bigintToI53Checked(instance);if(!("wgslLanguageFeatures"in navigator.gpu)){return false}return navigator.gpu.wgslLanguageFeatures.has(WebGPU.WGSLLanguageFeatureName[featureEnumValue])}var _wgpuQueueSubmit=function(queuePtr,commandCount,commands){queuePtr=bigintToI53Checked(queuePtr);commandCount=bigintToI53Checked(commandCount);commands=bigintToI53Checked(commands);var queue=WebGPU.getJsObject(queuePtr);var cmds=Array.from((growMemViews(),HEAP64).subarray(commands/8,(commands+commandCount*8)/8),id=>WebGPU.getJsObject(id));queue.submit(cmds)};function _wgpuQueueWriteBuffer(queuePtr,bufferPtr,bufferOffset,data,size){queuePtr=bigintToI53Checked(queuePtr);bufferPtr=bigintToI53Checked(bufferPtr);bufferOffset=bigintToI53Checked(bufferOffset);data=bigintToI53Checked(data);size=bigintToI53Checked(size);var queue=WebGPU.getJsObject(queuePtr);var buffer=WebGPU.getJsObject(bufferPtr);var subarray=(growMemViews(),HEAPU8).subarray(data,data+size);queue.writeBuffer(buffer,bufferOffset,subarray,0,size)}var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){let isAsyncifyImport=original.isAsync||importPattern.test(x);if(isAsyncifyImport){imports[x]=original=new WebAssembly.Suspending(original)}}}},instrumentFunction(original){var wrapper=(...args)=>original(...args);return wrapper},instrumentWasmExports(exports){var exportPattern=/^(wllama_start|wllama_action|main|__main_argc_argv)$/;Asyncify.asyncExports=new Set;var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){let isAsyncifyExport=exportPattern.test(x);if(isAsyncifyExport){Asyncify.asyncExports.add(original);original=Asyncify.makeAsyncFunction(original)}var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper}else{ret[x]=original}}return ret},asyncExports:null,isAsyncExport(func){return Asyncify.asyncExports?.has(func)},handleAsync:async startAsync=>{runtimeKeepalivePush();try{return await startAsync()}finally{runtimeKeepalivePop()}},handleSleep:startAsync=>Asyncify.handleAsync(()=>new Promise(startAsync)),makeAsyncFunction(original){return WebAssembly.promising(original)}};var getCFunc=ident=>{var func=Module["_"+ident];return func};var writeArrayToMemory=(array,buffer)=>{(growMemViews(),HEAP8).set(array,buffer)};var ccall=(ident,returnType,argTypes,args,opts)=>{var toC={pointer:p=>BigInt(p),string:str=>{var ret=0;if(str!==null&&str!==undefined&&str!==0){ret=stringToUTF8OnStack(str)}return BigInt(ret)},array:arr=>{var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return BigInt(ret)}};function convertReturnValue(ret){if(returnType==="string"){return UTF8ToString(Number(ret))}if(returnType==="pointer")return Number(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func(...cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}var asyncMode=opts?.async;if(asyncMode)return ret.then(onDone);ret=onDone(ret);return ret};var cwrap=(ident,returnType,argTypes,opts)=>{var numericArgs=!argTypes||argTypes.every(type=>type==="number"||type==="boolean");var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return(...args)=>ccall(ident,returnType,argTypes,args,opts)};var FS_createPath=(...args)=>FS.createPath(...args);var FS_unlink=(...args)=>FS.unlink(...args);var FS_createLazyFile=(...args)=>FS.createLazyFile(...args);var FS_createDevice=(...args)=>FS.createDevice(...args);PThread.init();FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{initMemory();if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}Module["ENV"]=ENV;Module["mmapAlloc"]=mmapAlloc;Module["wasmMemory"]=wasmMemory;Module["addRunDependency"]=addRunDependency;Module["removeRunDependency"]=removeRunDependency;Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["FS_preloadFile"]=FS_preloadFile;Module["FS_unlink"]=FS_unlink;Module["FS_createPath"]=FS_createPath;Module["FS_createDevice"]=FS_createDevice;Module["FS"]=FS;Module["FS_createDataFile"]=FS_createDataFile;Module["FS_createLazyFile"]=FS_createLazyFile;Module["MEMFS"]=MEMFS;var proxiedFunctionTable=[_proc_exit,exitOnMainThread,pthreadCreateProxied,___syscall_fcntl64,___syscall_fstat64,___syscall_getcwd,___syscall_getdents64,___syscall_ioctl,___syscall_lstat64,___syscall_newfstatat,___syscall_openat,___syscall_stat64,__mmap_js,__munmap_js,_environ_get,_environ_sizes_get,_fd_close,_fd_read,_fd_seek,_fd_write];function __asyncjs__js_file_read(path_ptr,offset,req_size,out_ptr){return Asyncify.handleAsync(async()=>await _wllama_js_file_read(UTF8ToString(Number(path_ptr)),Number(offset),Number(req_size),Number(out_ptr)))}__asyncjs__js_file_read.sig="jjjjj";var _wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug,_main,_malloc,_free,_emwgpuCreateBindGroup,_emwgpuCreateBindGroupLayout,_emwgpuCreateCommandBuffer,_emwgpuCreateCommandEncoder,_emwgpuCreateComputePassEncoder,_emwgpuCreateComputePipeline,_emwgpuCreateExternalTexture,_emwgpuCreatePipelineLayout,_emwgpuCreateQuerySet,_emwgpuCreateRenderBundle,_emwgpuCreateRenderBundleEncoder,_emwgpuCreateRenderPassEncoder,_emwgpuCreateRenderPipeline,_emwgpuCreateSampler,_emwgpuCreateSurface,_emwgpuCreateTexture,_emwgpuCreateTextureView,_emwgpuCreateAdapter,_emwgpuCreateBuffer,_emwgpuCreateDevice,_emwgpuCreateQueue,_emwgpuCreateShaderModule,_emwgpuOnDeviceLostCompleted,_emwgpuOnMapAsyncCompleted,_emwgpuOnRequestAdapterCompleted,_emwgpuOnRequestDeviceCompleted,_emwgpuOnWorkDoneCompleted,_emwgpuOnUncapturedError,__emscripten_tls_init,_pthread_self,_emscripten_builtin_memalign,__emscripten_thread_init,__emscripten_thread_crashed,__emscripten_run_js_on_main_thread,__emscripten_thread_free_data,__emscripten_thread_exit,__emscripten_check_mailbox,_memalign,___trap,_emscripten_stack_set_limits,__emscripten_stack_restore,__emscripten_stack_alloc,_emscripten_stack_get_current,__indirect_function_table,wasmTable;function assignWasmExports(wasmExports){_wllama_malloc=Module["_wllama_malloc"]=wasmExports["wllama_malloc"];_wllama_start=Module["_wllama_start"]=wasmExports["wllama_start"];_wllama_action=Module["_wllama_action"]=wasmExports["wllama_action"];_wllama_exit=Module["_wllama_exit"]=wasmExports["wllama_exit"];_wllama_debug=Module["_wllama_debug"]=wasmExports["wllama_debug"];_main=Module["_main"]=wasmExports["main"];_malloc=wasmExports["malloc"];_free=wasmExports["free"];_emwgpuCreateBindGroup=wasmExports["emwgpuCreateBindGroup"];_emwgpuCreateBindGroupLayout=wasmExports["emwgpuCreateBindGroupLayout"];_emwgpuCreateCommandBuffer=wasmExports["emwgpuCreateCommandBuffer"];_emwgpuCreateCommandEncoder=wasmExports["emwgpuCreateCommandEncoder"];_emwgpuCreateComputePassEncoder=wasmExports["emwgpuCreateComputePassEncoder"];_emwgpuCreateComputePipeline=wasmExports["emwgpuCreateComputePipeline"];_emwgpuCreateExternalTexture=wasmExports["emwgpuCreateExternalTexture"];_emwgpuCreatePipelineLayout=wasmExports["emwgpuCreatePipelineLayout"];_emwgpuCreateQuerySet=wasmExports["emwgpuCreateQuerySet"];_emwgpuCreateRenderBundle=wasmExports["emwgpuCreateRenderBundle"];_emwgpuCreateRenderBundleEncoder=wasmExports["emwgpuCreateRenderBundleEncoder"];_emwgpuCreateRenderPassEncoder=wasmExports["emwgpuCreateRenderPassEncoder"];_emwgpuCreateRenderPipeline=wasmExports["emwgpuCreateRenderPipeline"];_emwgpuCreateSampler=wasmExports["emwgpuCreateSampler"];_emwgpuCreateSurface=wasmExports["emwgpuCreateSurface"];_emwgpuCreateTexture=wasmExports["emwgpuCreateTexture"];_emwgpuCreateTextureView=wasmExports["emwgpuCreateTextureView"];_emwgpuCreateAdapter=wasmExports["emwgpuCreateAdapter"];_emwgpuCreateBuffer=wasmExports["emwgpuCreateBuffer"];_emwgpuCreateDevice=wasmExports["emwgpuCreateDevice"];_emwgpuCreateQueue=wasmExports["emwgpuCreateQueue"];_emwgpuCreateShaderModule=wasmExports["emwgpuCreateShaderModule"];_emwgpuOnDeviceLostCompleted=wasmExports["emwgpuOnDeviceLostCompleted"];_emwgpuOnMapAsyncCompleted=wasmExports["emwgpuOnMapAsyncCompleted"];_emwgpuOnRequestAdapterCompleted=wasmExports["emwgpuOnRequestAdapterCompleted"];_emwgpuOnRequestDeviceCompleted=wasmExports["emwgpuOnRequestDeviceCompleted"];_emwgpuOnWorkDoneCompleted=wasmExports["emwgpuOnWorkDoneCompleted"];_emwgpuOnUncapturedError=wasmExports["emwgpuOnUncapturedError"];__emscripten_tls_init=wasmExports["_emscripten_tls_init"];_pthread_self=wasmExports["pthread_self"];_emscripten_builtin_memalign=wasmExports["emscripten_builtin_memalign"];__emscripten_thread_init=wasmExports["_emscripten_thread_init"];__emscripten_thread_crashed=wasmExports["_emscripten_thread_crashed"];__emscripten_run_js_on_main_thread=wasmExports["_emscripten_run_js_on_main_thread"];__emscripten_thread_free_data=wasmExports["_emscripten_thread_free_data"];__emscripten_thread_exit=wasmExports["_emscripten_thread_exit"];__emscripten_check_mailbox=wasmExports["_emscripten_check_mailbox"];_memalign=wasmExports["memalign"];___trap=wasmExports["__trap"];_emscripten_stack_set_limits=wasmExports["emscripten_stack_set_limits"];__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"];__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"];__indirect_function_table=wasmTable=wasmExports["__indirect_function_table"]}var wasmImports;function assignWasmImports(){wasmImports={__asyncjs__js_file_read,__pthread_create_js:___pthread_create_js,__syscall_fcntl64:___syscall_fcntl64,__syscall_getcwd:___syscall_getcwd,__syscall_getdents64:___syscall_getdents64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,__syscall_stat64:___syscall_stat64,_abort_js:__abort_js,_emscripten_init_main_thread_js:__emscripten_init_main_thread_js,_emscripten_notify_mailbox_postmessage:__emscripten_notify_mailbox_postmessage,_emscripten_receive_on_main_thread_js:__emscripten_receive_on_main_thread_js,_emscripten_thread_cleanup:__emscripten_thread_cleanup,_emscripten_thread_mailbox_await:__emscripten_thread_mailbox_await,_emscripten_thread_set_strongref:__emscripten_thread_set_strongref,_localtime_js:__localtime_js,_mmap_js:__mmap_js,_munmap_js:__munmap_js,_tzset_js:__tzset_js,clock_time_get:_clock_time_get,emscripten_check_blocking_allowed:_emscripten_check_blocking_allowed,emscripten_date_now:_emscripten_date_now,emscripten_exit_with_live_runtime:_emscripten_exit_with_live_runtime,emscripten_get_callstack:_emscripten_get_callstack,emscripten_get_heap_max:_emscripten_get_heap_max,emscripten_get_now:_emscripten_get_now,emscripten_has_asyncify:_emscripten_has_asyncify,emscripten_num_logical_cores:_emscripten_num_logical_cores,emscripten_resize_heap:_emscripten_resize_heap,emwgpuAdapterRequestDevice:_emwgpuAdapterRequestDevice,emwgpuBufferDestroy:_emwgpuBufferDestroy,emwgpuBufferGetConstMappedRange:_emwgpuBufferGetConstMappedRange,emwgpuBufferMapAsync:_emwgpuBufferMapAsync,emwgpuBufferUnmap:_emwgpuBufferUnmap,emwgpuDelete:_emwgpuDelete,emwgpuDeviceCreateBuffer:_emwgpuDeviceCreateBuffer,emwgpuDeviceCreateShaderModule:_emwgpuDeviceCreateShaderModule,emwgpuDeviceDestroy:_emwgpuDeviceDestroy,emwgpuInstanceRequestAdapter:_emwgpuInstanceRequestAdapter,emwgpuQueueOnSubmittedWorkDone:_emwgpuQueueOnSubmittedWorkDone,emwgpuWaitAny:_emwgpuWaitAny,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,memory:wasmMemory,random_get:_random_get,wgpuAdapterGetInfo:_wgpuAdapterGetInfo,wgpuAdapterGetLimits:_wgpuAdapterGetLimits,wgpuAdapterHasFeature:_wgpuAdapterHasFeature,wgpuBufferGetSize:_wgpuBufferGetSize,wgpuCommandEncoderBeginComputePass:_wgpuCommandEncoderBeginComputePass,wgpuCommandEncoderCopyBufferToBuffer:_wgpuCommandEncoderCopyBufferToBuffer,wgpuCommandEncoderFinish:_wgpuCommandEncoderFinish,wgpuComputePassEncoderDispatchWorkgroups:_wgpuComputePassEncoderDispatchWorkgroups,wgpuComputePassEncoderEnd:_wgpuComputePassEncoderEnd,wgpuComputePassEncoderSetBindGroup:_wgpuComputePassEncoderSetBindGroup,wgpuComputePassEncoderSetPipeline:_wgpuComputePassEncoderSetPipeline,wgpuComputePipelineGetBindGroupLayout:_wgpuComputePipelineGetBindGroupLayout,wgpuDeviceCreateBindGroup:_wgpuDeviceCreateBindGroup,wgpuDeviceCreateCommandEncoder:_wgpuDeviceCreateCommandEncoder,wgpuDeviceCreateComputePipeline:_wgpuDeviceCreateComputePipeline,wgpuInstanceHasWGSLLanguageFeature:_wgpuInstanceHasWGSLLanguageFeature,wgpuQueueSubmit:_wgpuQueueSubmit,wgpuQueueWriteBuffer:_wgpuQueueWriteBuffer}}function applySignatureConversions(wasmExports){wasmExports=Object.assign({},wasmExports);var makeWrapper___PP=f=>(a0,a1,a2)=>f(a0,BigInt(a1?a1:0),BigInt(a2?a2:0));var makeWrapper_pp=f=>a0=>Number(f(BigInt(a0)));var makeWrapper__p=f=>a0=>f(BigInt(a0));var makeWrapper_p=f=>()=>Number(f());var makeWrapper_ppp=f=>(a0,a1)=>Number(f(BigInt(a0),BigInt(a1)));var makeWrapper__p_____=f=>(a0,a1,a2,a3,a4,a5)=>f(BigInt(a0),a1,a2,a3,a4,a5);var makeWrapper___p_p_=f=>(a0,a1,a2,a3,a4)=>f(a0,BigInt(a1),a2,BigInt(a3),a4);var makeWrapper__pp=f=>(a0,a1)=>f(BigInt(a0),BigInt(a1));wasmExports["main"]=makeWrapper___PP(wasmExports["main"]);wasmExports["malloc"]=makeWrapper_pp(wasmExports["malloc"]);wasmExports["free"]=makeWrapper__p(wasmExports["free"]);wasmExports["pthread_self"]=makeWrapper_p(wasmExports["pthread_self"]);wasmExports["emscripten_builtin_memalign"]=makeWrapper_ppp(wasmExports["emscripten_builtin_memalign"]);wasmExports["_emscripten_thread_init"]=makeWrapper__p_____(wasmExports["_emscripten_thread_init"]);wasmExports["_emscripten_run_js_on_main_thread"]=makeWrapper___p_p_(wasmExports["_emscripten_run_js_on_main_thread"]);wasmExports["_emscripten_thread_free_data"]=makeWrapper__p(wasmExports["_emscripten_thread_free_data"]);wasmExports["_emscripten_thread_exit"]=makeWrapper__p(wasmExports["_emscripten_thread_exit"]);wasmExports["memalign"]=makeWrapper_ppp(wasmExports["memalign"]);wasmExports["emscripten_stack_set_limits"]=makeWrapper__pp(wasmExports["emscripten_stack_set_limits"]);wasmExports["_emscripten_stack_restore"]=makeWrapper__p(wasmExports["_emscripten_stack_restore"]);wasmExports["_emscripten_stack_alloc"]=makeWrapper_pp(wasmExports["_emscripten_stack_alloc"]);wasmExports["emscripten_stack_get_current"]=makeWrapper_p(wasmExports["emscripten_stack_get_current"]);return wasmExports}async function callMain(){var entryFunction=_main;var argc=0;var argv=0;try{var ret=entryFunction(argc,BigInt(argv));ret=await ret;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(){if(runDependencies>0){dependenciesFulfilled=run;return}if(ENVIRONMENT_IS_PTHREAD){initRuntime();return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}async function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)await callMain();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}var wasmExports;if(!ENVIRONMENT_IS_PTHREAD){createWasm();run()}\n';

// src/worker.ts
var FILE_READ_REQ_EVENT = "fs.read_req";
var JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;
var ProxyToWorker = class {
  // filename -> Blob for async reads
  constructor(resources, nbThread, suppressNativeLog, logger) {
    __publicField(this, "resources");
    __publicField(this, "logger");
    __publicField(this, "suppressNativeLog");
    __publicField(this, "taskQueue", []);
    __publicField(this, "taskId", 1);
    __publicField(this, "resultQueue", []);
    __publicField(this, "busy", false);
    // is the work loop is running?
    __publicField(this, "worker");
    __publicField(this, "multiThread");
    __publicField(this, "nbThread");
    __publicField(this, "useAsyncFile");
    __publicField(this, "fileBlobs", /* @__PURE__ */ new Map());
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }
  getModuleCode() {
    return __async(this, null, function* () {
      if (!this.resources.jsPath) {
        if (this.resources.compat) {
          throw new Error(
            "compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat."
          );
        }
        return WLLAMA_EMSCRIPTEN_CODE;
      } else if (this.resources.jsPath.code) {
        return this.resources.jsPath.code;
      } else if (isString(this.resources.jsPath)) {
        const response = yield fetch(this.resources.jsPath);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch worker code from ${this.resources.jsPath}`
          );
        }
        return yield response.text();
      } else {
        throw new Error("No JS code provided for worker");
      }
    });
  }
  moduleInit(ggufFiles) {
    return __async(this, null, function* () {
      let moduleCode = JSPI_STUB + (yield this.getModuleCode());
      if (this.resources.noWebGPU) {
        moduleCode = 'try{Object.defineProperty(WorkerNavigator.prototype,"gpu",{get:()=>({requestAdapter:async()=>null})});}catch(e){}' + moduleCode;
      }
      let mainModuleCode = moduleCode.replace("var Module", "var ___Module");
      const runOptions = {
        pathConfig: {
          "wllama.wasm": this.resources.wasmPath
        },
        nbThread: this.nbThread,
        compat: this.resources.compat
      };
      const completeCode = [
        `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
        `function wModuleInit() { ${mainModuleCode}; return Module; }`,
        LLAMA_CPP_WORKER_CODE
      ].join(";\n\n");
      this.worker = createWorker(completeCode);
      this.worker.onmessage = this.onRecvMsg.bind(this);
      this.worker.onerror = this.logger.error;
      const res = yield this.pushTask({
        verb: "module.init",
        args: [
          new Blob([moduleCode], { type: "text/javascript" }),
          this.useAsyncFile
        ],
        callbackId: this.taskId++
      });
      const nativeFiles = [];
      for (const file of ggufFiles) {
        const needAllocBuffer = !this.useAsyncFile;
        const id = yield this.fileAlloc(
          file.name,
          file.blob.size,
          needAllocBuffer
        );
        nativeFiles.push(__spreadValues({ id }, file));
        if (this.useAsyncFile) {
          this.fileBlobs.set(file.name, file.blob);
        }
      }
      if (!this.useAsyncFile) {
        yield Promise.all(
          nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
          })
        );
      }
      return res;
    });
  }
  wllamaStart() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.start",
        args: [],
        callbackId: this.taskId++
      });
      const parsedResult = this.parseResult(result);
      return parsedResult;
    });
  }
  wllamaAction(name, body) {
    return __async(this, null, function* () {
      const encodedMsg = glueSerialize(body);
      const result = yield this.pushTask({
        verb: "wllama.action",
        args: [name, encodedMsg],
        callbackId: this.taskId++
      });
      const parsedResult = glueDeserialize(result);
      if (parsedResult._name === "erro_evt") {
        throw new WllamaRuntimeError(parsedResult.message, "");
      }
      return parsedResult;
    });
  }
  /** Take a slot save file out of the worker's in-memory FS (the file is removed) */
  slotFileRead(fileName) {
    return __async(this, null, function* () {
      return yield this.pushTask({
        verb: "fs.slot_read",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  /** Put a slot save file into the worker's in-memory FS */
  slotFileWrite(fileName, data) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_write",
        args: [fileName, data],
        callbackId: this.taskId++
      });
    });
  }
  slotFileRemove(fileName) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_remove",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  wllamaExit() {
    return __async(this, null, function* () {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  wllamaDebug() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.debug",
        args: [],
        callbackId: this.taskId++
      });
      return JSON.parse(result);
    });
  }
  ///////////////////////////////////////
  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  fileAlloc(fileName, size, allocBuffer) {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "fs.alloc",
        args: [fileName, size, allocBuffer],
        callbackId: this.taskId++
      });
      return result.fileId;
    });
  }
  /**
   * Write a Blob to heapfs
   */
  fileWrite(fileId, blob) {
    return __async(this, null, function* () {
      const reader = blob.stream().getReader();
      let offset = 0;
      while (true) {
        const { done, value } = yield reader.read();
        if (done) break;
        const size = value.byteLength;
        yield this.pushTask(
          {
            verb: "fs.write",
            args: [fileId, value, offset],
            callbackId: this.taskId++
          },
          // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
          [value.buffer]
        );
        offset += size;
      }
    });
  }
  fileReadResponse(name, offset, size) {
    return __async(this, null, function* () {
      var _a;
      try {
        const blob = this.fileBlobs.get(name);
        if (!blob) {
          throw new Error(`blob not found for name="${name}"`);
        }
        const chunk = blob.slice(offset, offset + size);
        const buffer = yield chunk.arrayBuffer();
        this.worker.postMessage(
          { verb: "fs.read_res", args: [buffer] },
          { transfer: [buffer] }
        );
      } catch (err) {
        this.logger.error("fileReadResponse failed, terminating worker:", err);
        (_a = this.worker) == null ? void 0 : _a.terminate();
        this.worker = void 0;
        this.abort(`File read failed: ${err}`, err.stack || "");
      }
    });
  }
  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  parseResult(result) {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult["error"]) {
      throw new WllamaRuntimeError("Unknown error, please see console.log", "");
    }
    return parsedResult;
  }
  /**
   * Push a new task to taskQueue
   */
  pushTask(param, buffers) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }
  /**
   * Main loop for processing tasks
   */
  runTaskLoop() {
    return __async(this, null, function* () {
      var _a;
      if (this.busy) {
        return;
      }
      this.busy = true;
      while (true) {
        const task = this.taskQueue.shift();
        if (!task) break;
        this.resultQueue.push(task);
        this.worker.postMessage(
          task.param,
          isSafariMobile() ? void 0 : {
            transfer: (_a = task.buffers) != null ? _a : []
          }
        );
      }
      this.busy = false;
    });
  }
  /**
   * Handle messages from worker
   */
  onRecvMsg(e) {
    if (!e.data) return;
    const { verb, args } = e.data;
    const isCompatBuild = this.resources.compat;
    if (verb && verb.startsWith("console.")) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith("debug")) this.logger.debug(...args);
      if (verb.endsWith("log")) this.logger.log(...args);
      if (verb.endsWith("warn")) this.logger.warn(...args);
      if (verb.endsWith("error")) this.logger.error(...args);
      return;
    } else if (verb === "signal.abort") {
      const [signalType, message, rawStack, originalErr] = args;
      if (originalErr) {
        this.logger.error(originalErr);
      }
      (() => __async(this, null, function* () {
        let stack = "";
        let newMsg = message.replace(
          "Build with -sASSERTIONS for more info.",
          ""
        );
        if (signalType === "abort") {
          newMsg = `(ABORT) ${newMsg}`;
          stack = rawStack.replace(/\|/g, "\n");
        } else if (signalType === "exception") {
          stack = rawStack;
        }
        const decoded = yield Debug.decodeStackTrace(stack, isCompatBuild);
        this.logger.error(`Stack trace (${signalType}):
` + decoded);
        this.abort(newMsg, decoded);
      }))();
      return;
    }
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args;
      this.fileReadResponse(name, offset, size).catch(() => {
      });
      return;
    }
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }
  abort(text, stack) {
    const error = new WllamaRuntimeError(
      text.length == 0 ? "(unknown error)" : text,
      stack
    );
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(error);
    }
    while (this.taskQueue.length > 0) {
      const pendingTask = this.taskQueue.pop();
      if (!pendingTask) break;
      pendingTask.reject(error);
    }
  }
};

// src/huggingface.ts
var HF_BASE = "https://huggingface.co";
var DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"];
function fetchRepoFiles(repo, token) {
  return __async(this, null, function* () {
    var _a;
    const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
    const headers = { Accept: "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = yield fetch(url, { headers });
    if (!res.ok) {
      let msg = res.statusText;
      try {
        msg = (_a = (yield res.json()).error) != null ? _a : msg;
      } catch (e) {
      }
      throw new Error(`HF API error (${res.status}): ${msg}`);
    }
    return res.json();
  });
}
function firstShardPath(files, path) {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}
function selectFile(files, quant, mmprojOnly) {
  const candidates = files.filter((f) => {
    if (f.type !== "file" || !f.path.toLowerCase().endsWith(".gguf"))
      return false;
    const ismmproj = f.path.toLowerCase().includes("mmproj");
    return mmprojOnly ? ismmproj : !ismmproj;
  });
  if (candidates.length === 0) return null;
  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }
  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }
  return firstShardPath(candidates, candidates[0].path);
}
function getHFModelSource(config) {
  return __async(this, null, function* () {
    const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;
    const files = yield fetchRepoFiles(repo, hfToken);
    const modelPath = file != null ? file : selectFile(files, quant, false);
    if (!modelPath) {
      throw new Error(`No GGUF file found in repo "${repo}"`);
    }
    const source = {
      url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`
    };
    if (mmprojFile || mmprojQuant !== void 0) {
      const mmpath = mmprojFile != null ? mmprojFile : selectFile(files, mmprojQuant, true);
      if (mmpath) {
        source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
      }
    }
    if (hfToken) {
      const params = new URLSearchParams({ token: hfToken });
      source.url += `?${params}`;
      if (source.mmprojUrl) {
        source.mmprojUrl += `?${params}`;
      }
    }
    return source;
  });
}
function getHFFileSHA256(url, headers) {
  return __async(this, null, function* () {
    if (!url.includes("/resolve/")) return void 0;
    const rawUrl = url.replace("/resolve/", "/raw/");
    try {
      const text = yield fetch(rawUrl, { headers }).then((r) => r.text());
      const match = text.match(/^oid sha256:([0-9a-f]{64})$/m);
      return match ? match[1] : void 0;
    } catch (e) {
      return void 0;
    }
  });
}

// src/storage/opfs.ts
var OPFSBackend = class {
  isSupported() {
    var _a;
    return typeof navigator !== "undefined" && "storage" in navigator && !!((_a = navigator.storage) == null ? void 0 : _a.getDirectory);
  }
  read(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        return yield fileHandle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  write(key, stream) {
    return __async(this, null, function* () {
      const writable = yield openWritable(key);
      yield writable.truncate(0);
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        const file = yield fileHandle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      const cacheDir = yield getCacheDir();
      const result = [];
      try {
        for (var iter = __forAwait(cacheDir.entries()), more, temp, error; more = !(temp = yield iter.next()).done; more = false) {
          const [name, handle] = temp.value;
          if (handle.kind === "file") {
            const file = yield handle.getFile();
            result.push({ key: name, size: file.size });
          }
        }
      } catch (temp) {
        error = [temp];
      } finally {
        try {
          more && (temp = iter.return) && (yield temp.call(iter));
        } finally {
          if (error)
            throw error[0];
        }
      }
      return result;
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        yield cacheDir.removeEntry(key);
      } catch (e) {
        if ((e == null ? void 0 : e.name) !== "NotFoundError") throw e;
      }
    });
  }
};
function getCacheDir() {
  return __async(this, null, function* () {
    const opfsRoot = yield navigator.storage.getDirectory();
    return opfsRoot.getDirectoryHandle("cache", { create: true });
  });
}
function openWritable(fileName) {
  return __async(this, null, function* () {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let pResolve;
    let pReject;
    worker.onmessage = (e) => {
      if (e.data.ok) pResolve(null);
      else if (e.data.err) pReject(e.data.err);
    };
    worker.onerror = (e) => {
      var _a;
      return pReject == null ? void 0 : pReject((_a = e.message) != null ? _a : e);
    };
    const workerExec = (data) => new Promise((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile() ? void 0 : { transfer: "buf" in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
    yield workerExec({ action: "open", filename: fileName });
    return {
      truncate: () => __async(this, null, function* () {
      }),
      write: (value) => workerExec({ action: "write", buf: value }),
      close: () => __async(this, null, function* () {
        yield workerExec({ action: "close" });
        worker.terminate();
      })
    };
  });
}

// src/storage/cos.ts
function makeHash(key) {
  return { algorithm: "SHA-256", value: key };
}
var COSInternalBackend = class {
  isSupported() {
    return typeof navigator !== "undefined" && "crossOriginStorage" in navigator;
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  read(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        return handle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  write(key, stream) {
    return __async(this, null, function* () {
      const handle = yield navigator.crossOriginStorage.requestFileHandle(
        makeHash(key),
        { create: true }
      );
      const writable = yield handle.createWritable();
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        const file = yield handle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
  delete(_key) {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
};
var COSBackend = class {
  constructor() {
    __publicField(this, "cos", new COSInternalBackend());
    __publicField(this, "priv", new OPFSBackend());
  }
  isSupported() {
    return this.priv.isSupported();
  }
  read(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const blob = yield this.cos.read(hint.sha256);
        if (blob) return blob;
      }
      return this.priv.read(key);
    });
  }
  write(key, stream, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        yield this.cos.write(hint.sha256, stream);
      } else {
        yield this.priv.write(key, stream);
      }
    });
  }
  getSize(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const size = yield this.cos.getSize(hint.sha256);
        if (size !== -1) return size;
      }
      return this.priv.getSize(key);
    });
  }
  list() {
    return __async(this, null, function* () {
      return this.priv.list();
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      return this.priv.delete(key);
    });
  }
};

// src/cache-manager.ts
var PREFIX_METADATA = "__metadata__";
var POLYFILL_ETAG = "polyfill_for_older_version";
function hintFromMetadata(metadata) {
  if (!metadata) return void 0;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return void 0;
}
var CacheManager = class {
  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends = [new COSBackend()]) {
    __publicField(this, "sb");
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error("No supported storage backend found");
  }
  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  getNameFromURL(url) {
    return __async(this, null, function* () {
      return urlToFileName(url, "");
    });
  }
  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  write(name, stream, metadata) {
    return __async(this, null, function* () {
      yield this.sb.write(name, stream);
      yield this.writeMetadata(name, metadata);
    });
  }
  download(_0) {
    return __async(this, arguments, function* (url, options = {}) {
      var _a, _b, _c, _d;
      const fileKey = yield urlToFileName(url, "");
      const sha256 = yield getHFFileSHA256(url, (_a = options.headers) != null ? _a : {});
      const hint = sha256 ? { sha256 } : void 0;
      const cachedSize = yield this.sb.getSize(fileKey, hint);
      if (cachedSize !== -1) {
        const metadata2 = yield this.readMetadata(fileKey);
        if ((metadata2 == null ? void 0 : metadata2.originalURL) === url && metadata2.originalSize === cachedSize) {
          return;
        }
        const head = yield fetch(url, __spreadValues({
          method: "HEAD"
        }, options.headers ? { headers: options.headers } : {}));
        const originalSize = parseInt(
          (_b = head.headers.get("content-length")) != null ? _b : "0",
          10
        );
        const etag2 = (head.headers.get("etag") || "").replace(
          /[^A-Za-z0-9]/g,
          ""
        );
        if (originalSize > 0 && originalSize === cachedSize) {
          yield this.writeMetadata(fileKey, __spreadValues({
            originalURL: url,
            originalSize,
            etag: etag2,
            sha256
          }, (_c = options.metadataAdditional) != null ? _c : {}));
          return;
        }
        yield this.sb.delete(fileKey);
        yield this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
      const response = yield fetch(url, __spreadValues(__spreadValues({}, options.headers ? { headers: options.headers } : {}), options.signal ? { signal: options.signal } : {}));
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
      }
      const contentLength = response.headers.get("content-length");
      const etag = (response.headers.get("etag") || "").replace(
        /[^A-Za-z0-9]/g,
        ""
      );
      const total = parseInt(contentLength != null ? contentLength : "0", 10);
      const progressCallback = options.progressCallback;
      let loaded = 0;
      let lastProgressAt = 0;
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (progressCallback) {
            const now = Date.now();
            if (now - lastProgressAt > 100) {
              lastProgressAt = now;
              progressCallback({ loaded, total });
            }
          }
          controller.enqueue(chunk);
        },
        flush() {
          progressCallback == null ? void 0 : progressCallback({ loaded, total: total || loaded });
        }
      });
      const metadata = __spreadValues({
        originalURL: url,
        originalSize: total,
        etag
      }, (_d = options.metadataAdditional) != null ? _d : {});
      if (sha256) {
        metadata.sha256 = sha256;
      }
      yield this.sb.write(
        fileKey,
        response.body.pipeThrough(progressStream),
        hint
      );
      yield this.writeMetadata(fileKey, metadata);
    });
  }
  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  open(nameOrURL) {
    return __async(this, null, function* () {
      const hint1 = hintFromMetadata(yield this.getMetadata(nameOrURL));
      const direct = yield this.sb.read(nameOrURL, hint1);
      if (direct) return direct;
      const key = yield urlToFileName(nameOrURL, "");
      const hint2 = hintFromMetadata(yield this.getMetadata(key));
      return this.sb.read(key, hint2);
    });
  }
  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  getSize(name) {
    return __async(this, null, function* () {
      const hint = hintFromMetadata(yield this.getMetadata(name));
      return this.sb.getSize(name, hint);
    });
  }
  /**
   * Get metadata of a cached file
   */
  getMetadata(name) {
    return __async(this, null, function* () {
      const metadata = yield this.readMetadata(name);
      if (metadata) return metadata;
      const cachedSize = yield this.sb.getSize(name);
      return cachedSize > 0 ? (
        // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: ""
        }
      ) : (
        // cached file not found
        null
      );
    });
  }
  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  readMetadata(name) {
    return __async(this, null, function* () {
      const blob = yield this.sb.read(`${PREFIX_METADATA}${name}`);
      if (!blob) return null;
      try {
        return yield new Response(blob).json();
      } catch (e) {
        return null;
      }
    });
  }
  /**
   * List all files currently in cache
   */
  list() {
    return __async(this, null, function* () {
      const all = yield this.sb.list();
      const metadataMap = {};
      for (const { key } of all) {
        if (key.startsWith(PREFIX_METADATA)) {
          const blob = yield this.sb.read(key);
          if (blob) {
            const meta = yield new Response(blob).json().catch(() => null);
            metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
          }
        }
      }
      const result = [];
      for (const { key, size } of all) {
        if (!key.startsWith(PREFIX_METADATA)) {
          result.push({
            name: key,
            size,
            metadata: metadataMap[key] || {
              originalSize: size,
              originalURL: "",
              etag: ""
            }
          });
        }
      }
      return result;
    });
  }
  /**
   * Clear all files currently in cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.deleteMany(() => true);
    });
  }
  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  delete(nameOrURL) {
    return __async(this, null, function* () {
      const name2 = yield this.getNameFromURL(nameOrURL);
      yield this.deleteMany(
        (entry) => entry.name === nameOrURL || entry.name === name2
      );
    });
  }
  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  deleteMany(predicate) {
    return __async(this, null, function* () {
      const list = yield this.list();
      for (const item of list) {
        if (predicate(item)) {
          yield this.sb.delete(item.name);
          yield this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        }
      }
    });
  }
  /**
   * Write the metadata of the file to disk.
   */
  writeMetadata(name, metadata) {
    return __async(this, null, function* () {
      const blob = new Blob([JSON.stringify(metadata)], { type: "text/plain" });
      yield this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
    });
  }
};
var cache_manager_default = CacheManager;
function urlToFileName(url, prefix) {
  return __async(this, null, function* () {
    const hashBuffer = yield crypto.subtle.digest(
      "SHA-1",
      new TextEncoder().encode(url)
    );
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `${prefix}${hashHex}_${url.split("/").pop()}`;
  });
}

// src/model-manager.ts
var DEFAULT_PARALLEL_DOWNLOADS = 3;
var ModelValidationStatus = /* @__PURE__ */ ((ModelValidationStatus2) => {
  ModelValidationStatus2["VALID"] = "valid";
  ModelValidationStatus2["INVALID"] = "invalid";
  ModelValidationStatus2["DELETED"] = "deleted";
  return ModelValidationStatus2;
})(ModelValidationStatus || {});
var Model = class {
  constructor(modelManager, url, mmprojUrl, savedFiles) {
    __publicField(this, "modelManager");
    /**
     * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
     *
     * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
     */
    __publicField(this, "url");
    /**
     * URL to mmproj file, if exists
     */
    __publicField(this, "mmprojUrl");
    /**
     * Size in bytes (total size of all shards).
     *
     * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
     */
    __publicField(this, "size");
    /**
     * List of all shards in the cache, sorted by original URL (ascending order)
     */
    __publicField(this, "files");
    this.modelManager = modelManager;
    this.url = url;
    this.mmprojUrl = mmprojUrl;
    if (savedFiles) {
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * Open and get a list of all shards as Blobs
   */
  open() {
    return __async(this, null, function* () {
      if (this.size === -1) {
        throw new WllamaError(
          `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
          "load_error"
        );
      }
      const blobs = [];
      for (const file of this.files) {
        const blob = yield this.modelManager.cacheManager.open(file.name);
        if (!blob) {
          throw new Error(
            `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
          );
        }
        blobs.push(blob);
      }
      return blobs;
    });
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate() {
    let nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.mmprojUrl) {
      nbShards += 1;
    }
    if (this.size === -1) {
      return "deleted" /* DELETED */;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return "invalid" /* INVALID */;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return "invalid" /* INVALID */;
      }
    }
    return "valid" /* VALID */;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  refresh() {
    return __async(this, arguments, function* (options = {}) {
      var _a;
      const urls = ModelManager.parseModelUrl(this.url);
      if (this.mmprojUrl) {
        urls.push(this.mmprojUrl);
      }
      const works = urls.map((url, index) => ({
        url,
        index
      }));
      this.modelManager.logger.debug("Downloading model files:", urls);
      const nParallel = (_a = this.modelManager.params.parallelDownloads) != null ? _a : DEFAULT_PARALLEL_DOWNLOADS;
      const totalSize = yield this.getTotalDownloadSize(urls);
      const loadedSize = [];
      const worker = () => __async(this, null, function* () {
        while (works.length > 0) {
          const w = works.shift();
          if (!w) break;
          yield this.modelManager.cacheManager.download(w.url, __spreadProps(__spreadValues({}, options), {
            metadataAdditional: {
              originalURL: w.url,
              mmprojURL: this.mmprojUrl
            },
            progressCallback: ({ loaded }) => {
              var _a2;
              loadedSize[w.index] = loaded;
              (_a2 = options.progressCallback) == null ? void 0 : _a2.call(options, {
                loaded: sumArr(loadedSize),
                total: totalSize
              });
            }
          }));
        }
      });
      const promises = [];
      for (let i = 0; i < nParallel; i++) {
        promises.push(worker());
        loadedSize.push(0);
      }
      yield Promise.all(promises);
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
    });
  }
  /**
   * Remove the model from the cache
   */
  remove() {
    return __async(this, null, function* () {
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      yield this.modelManager.cacheManager.deleteMany(
        (f) => !!this.files.find((file) => file.name === f.name)
      );
      this.size = -1;
    });
  }
  getAllFiles(savedFiles) {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    if (this.mmprojUrl) {
      allUrls.add(this.mmprojUrl);
    }
    const allFiles = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort(
      (a, b) => a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }
  getTotalDownloadSize(urls) {
    return __async(this, null, function* () {
      const responses = yield Promise.all(
        urls.map((url) => fetch(url, { method: "HEAD" }))
      );
      const sizes = responses.map(
        (res) => Number(res.headers.get("content-length") || "0")
      );
      return sumArr(sizes);
    });
  }
};
var ModelManager = class _ModelManager {
  constructor(params = {}) {
    // The CacheManager singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "params");
    __publicField(this, "logger");
    this.cacheManager = params.cacheManager || new cache_manager_default();
    this.params = params;
    this.logger = params.logger || console;
  }
  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl) {
    var _a;
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = (_a = queryMatch == null ? void 0 : queryMatch[1]) != null ? _a : "";
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, "");
    const total = matches[2];
    const paddedShardIds = Array.from(
      { length: Number(total) },
      (_, index) => (index + 1).toString().padStart(5, "0")
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }
  /**
   * Get all models in the cache
   */
  getModels() {
    return __async(this, arguments, function* (opts = {}) {
      const cachedFiles = yield this.cacheManager.list();
      let models = [];
      for (const file of cachedFiles) {
        if (!file.metadata.originalURL) continue;
        const shards = _ModelManager.parseModelUrl(file.metadata.originalURL);
        const mmprojUrl = file.metadata.mmprojURL;
        const isFirstShard = shards.length === 1 || shards[0] === file.metadata.originalURL;
        if (isFirstShard) {
          models.push(
            new Model(this, file.metadata.originalURL, mmprojUrl, cachedFiles)
          );
        }
      }
      if (!opts.includeInvalid) {
        models = models.filter(
          (m) => m.validate() === "valid" /* VALID */
        );
      }
      return models;
    });
  }
  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  downloadModel(_0) {
    return __async(this, arguments, function* (sourceOrURL, options = {}) {
      const source = isString(sourceOrURL) ? { url: sourceOrURL } : sourceOrURL;
      if (!isValidGgufFile(source.url)) {
        throw new WllamaError(
          `Invalid model URL: ${source.url}; URL must ends with ".gguf"`,
          "download_error"
        );
      }
      const model = new Model(this, source.url, source.mmprojUrl);
      const validity = model.validate();
      if (validity !== "valid" /* VALID */) {
        yield model.refresh(options);
      }
      return model;
    });
  }
  /**
   * Get a model from the cache or download it if it's not available.
   */
  getModelOrDownload(_0) {
    return __async(this, arguments, function* (source, options = {}) {
      var _a;
      const models = yield this.getModels();
      const model = models.find((m) => m.url === source.url);
      if (model) {
        (_a = options.progressCallback) == null ? void 0 : _a.call(options, { loaded: model.size, total: model.size });
        return model;
      }
      return this.downloadModel(source, options);
    });
  }
  /**
   * Remove all models from the cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.cacheManager.clear();
    });
  }
};

// src/types/types.ts
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2[LogLevel2["DEBUG"] = 1] = "DEBUG";
  LogLevel2[LogLevel2["INFO"] = 2] = "INFO";
  LogLevel2[LogLevel2["WARN"] = 3] = "WARN";
  LogLevel2[LogLevel2["ERROR"] = 4] = "ERROR";
  return LogLevel2;
})(LogLevel || {});

// src/wasm-from-cdn.ts
var WasmCompatFromCDN = {
  worker: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.js",
  wasm: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.wasm"
};

// src/wllama.ts
var LoggerWithoutDebug = __spreadProps(__spreadValues({}, console), {
  debug: () => {
  }
});
var WllamaError = class extends Error {
  constructor(message, type = "unknown_error") {
    super(message);
    __publicField(this, "type");
    this.type = type;
  }
};
var WllamaAbortError = class extends Error {
  constructor() {
    super("Operation aborted");
    __publicField(this, "name", "AbortError");
  }
};
var WllamaRuntimeError = class extends Error {
  constructor(message, stack) {
    super(message);
    __publicField(this, "name", "RuntimeError");
    __publicField(this, "stack");
    this.stack = stack;
  }
};
var Wllama = class {
  constructor(pathConfig, wllamaConfig = {}) {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "modelManager");
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs, `nativeStates` means `native_states` loads natlang native state spans, `natlangScore` means `createNatlangScore` is available, `slotState` means `saveSlot` / `restoreSlot` / `eraseSlot` / `applyTemplate` are available and completions accept `id_slot`, `nativeStateStore` means `putNativeState` / `getNativeState` are available. */
    __publicField(this, "capabilities", {
      loraBlobs: true,
      nativeStates: true,
      natlangScore: true,
      slotState: true,
      nativeStateStore: true
    });
    __publicField(this, "slotFileId", 0);
    __publicField(this, "compat", null);
    __publicField(this, "proxy", null);
    __publicField(this, "config");
    __publicField(this, "pathConfig");
    __publicField(this, "useMultiThread", false);
    __publicField(this, "nbThreads", 1);
    __publicField(this, "useEmbeddings", false);
    __publicField(this, "useRerank", false);
    // available when loaded
    __publicField(this, "loadedContextInfo", null);
    __publicField(this, "seed");
    __publicField(this, "bosToken", -1);
    __publicField(this, "eosToken", -1);
    __publicField(this, "eotToken", -1);
    __publicField(this, "eogTokens", /* @__PURE__ */ new Set());
    __publicField(this, "addBosToken", false);
    __publicField(this, "addEosToken", false);
    __publicField(this, "mediaMarker");
    __publicField(this, "chatTemplate");
    __publicField(this, "metadata");
    __publicField(this, "hasEncoder", false);
    __publicField(this, "decoderStartToken", -1);
    // the defaults are also the server's own (default_template_kwargs_keys/vals at load, as JSON texts), which is what
    // scoring renders with; the overlay keeps them on chat requests, whose own kwargs win
    __publicField(this, "chatTemplateKwargs", {});
    var _a, _b, _c;
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError("AssetsPathConfig is required");
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = (_a = wllamaConfig.cacheManager) != null ? _a : new cache_manager_default();
    this.modelManager = (_c = wllamaConfig.modelManager) != null ? _c : new ModelManager({
      cacheManager: this.cacheManager,
      logger: (_b = wllamaConfig.logger) != null ? _b : console,
      parallelDownloads: wllamaConfig.parallelDownloads,
      allowOffline: wllamaConfig.allowOffline
    });
    this.setCompat("default");
  }
  logger() {
    var _a;
    return (_a = this.config.logger) != null ? _a : console;
  }
  checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        "loadModel() is not yet called",
        "model_not_loaded"
      );
    }
  }
  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion() {
    return LIBLLAMA_VERSION;
  }
  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(compat, mode = "safari") {
    if (mode === "safari") {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === "default" ? WasmCompatFromCDN : compat;
  }
  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded() {
    return !!this.proxy && !!this.metadata;
  }
  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS() {
    return this.bosToken;
  }
  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS() {
    return this.eosToken;
  }
  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT() {
    return this.eotToken;
  }
  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token) {
    return token === this.eosToken || token === this.eotToken || this.eogTokens.has(token);
  }
  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken() {
    return this.decoderStartToken;
  }
  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata() {
    this.checkModelLoaded();
    return this.metadata;
  }
  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread() {
    this.checkModelLoaded();
    return this.useMultiThread;
  }
  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads() {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }
  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture() {
    this.checkModelLoaded();
    return this.hasEncoder;
  }
  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken() {
    this.checkModelLoaded();
    return this.addBosToken;
  }
  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken() {
    this.checkModelLoaded();
    return this.addEosToken;
  }
  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate() {
    var _a;
    this.checkModelLoaded();
    return (_a = this.chatTemplate) != null ? _a : null;
  }
  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU() {
    return isSupportWebGPU();
  }
  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  loadModelFromUrl(_0) {
    return __async(this, arguments, function* (modelSourceOrURL, params = {}) {
      var _a;
      const source = isString(modelSourceOrURL) ? { url: modelSourceOrURL } : modelSourceOrURL;
      const useCache = (_a = params.useCache) != null ? _a : true;
      const model = useCache ? yield this.modelManager.getModelOrDownload(source, params) : yield this.modelManager.downloadModel(source, params);
      const blobs = yield model.open();
      return yield this.loadModel(blobs, params);
    });
  }
  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  loadModelFromHF(_0) {
    return __async(this, arguments, function* (hfOptions, params = {}) {
      const source = yield getHFModelSource(hfOptions);
      return yield this.loadModelFromUrl(source, params);
    });
  }
  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  loadModel(_0) {
    return __async(this, arguments, function* (ggufBlobsOrModel, params = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k;
      const blobs = ggufBlobsOrModel instanceof Model ? yield ggufBlobsOrModel.open() : [...ggufBlobsOrModel];
      if (blobs.some((b) => b.size === 0)) {
        throw new WllamaError(
          "Input model (or splits) must be non-empty Blob or File",
          "load_error"
        );
      }
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (this.proxy) {
        throw new WllamaError("Module is already initialized", "load_error");
      }
      const supportMultiThread = yield isSupportMultiThread();
      const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
      const nbThreads = (_a = params.n_threads) != null ? _a : hwConccurency;
      this.nbThreads = nbThreads;
      this.useMultiThread = supportMultiThread && nbThreads > 1;
      const workerResources = this.getWorkerResources();
      if (params.n_gpu_layers === 0) {
        workerResources.noWebGPU = true;
      }
      this.proxy = new ProxyToWorker(
        workerResources,
        this.useMultiThread ? nbThreads : 0,
        // 0 means disable pthread
        (_b = this.config.suppressNativeLog) != null ? _b : false,
        this.logger()
      );
      let logLevel = (_c = params.log_level) != null ? _c : 2 /* INFO */;
      if (this.config.suppressNativeLog) {
        logLevel = 9999;
      }
      const modelFiles = yield prepareBlobs(blobs);
      const loraPaths = [];
      const sideFiles = [];
      for (const [i, adapter] of ((_d = params.lora_adapters) != null ? _d : []).entries()) {
        if (adapter.blob) {
          if (adapter.blob.size === 0) {
            throw new WllamaError(
              `LoRA adapter ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `lora-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: adapter.blob });
          loraPaths.push(`models/${name}`);
        } else if (adapter.path) {
          loraPaths.push(adapter.path);
        } else {
          throw new WllamaError(
            `LoRA adapter ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      const statePaths = [];
      for (const [i, state] of ((_e = params.native_states) != null ? _e : []).entries()) {
        if (state.blob) {
          if (state.blob.size === 0) {
            throw new WllamaError(
              `Native state ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `native-state-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: state.blob });
          statePaths.push(`models/${name}`);
        } else if (state.path) {
          statePaths.push(state.path);
        } else {
          throw new WllamaError(
            `Native state ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      yield this.proxy.moduleInit([...modelFiles.all, ...sideFiles]);
      this.logger().debug("Calling wllamaStart...");
      const startResult = yield this.proxy.wllamaStart();
      if (!startResult.success) {
        throw new WllamaError(
          `Error while calling start function, result = ${startResult}`
        );
      }
      this.logger().debug("Loading model...");
      const loadResult = yield this.proxy.wllamaAction("load", {
        _name: "load_req",
        log_level: logLevel,
        // if async read is not supported, use mmap; refer to README-dev.md for more details
        use_mmap: !canUseAsyncFileRead(workerResources.compat),
        use_mlock: false,
        n_gpu_layers: (_f = params.n_gpu_layers) != null ? _f : 99999,
        n_ctx: (_g = params.n_ctx) != null ? _g : 1024,
        n_threads: this.useMultiThread ? nbThreads : 1,
        n_ctx_auto: false,
        // not supported for now
        mmproj_path: modelFiles.mmproj ? `/models/${MMPROJ_FILE_NAME}` : void 0,
        model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
        embeddings: params.embeddings,
        offload_kqv: params.offload_kqv,
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        pooling_type: params.pooling_type,
        rope_scaling_type: params.rope_scaling_type,
        rope_freq_base: params.rope_freq_base,
        rope_freq_scale: params.rope_freq_scale,
        yarn_ext_factor: params.yarn_ext_factor,
        yarn_attn_factor: params.yarn_attn_factor,
        yarn_beta_fast: params.yarn_beta_fast,
        yarn_beta_slow: params.yarn_beta_slow,
        yarn_orig_ctx: params.yarn_orig_ctx,
        cache_type_k: params.cache_type_k,
        cache_type_v: params.cache_type_v,
        // with unified KV, all sequences share one n_ctx cache, so each request can still use the full context
        n_parallel: (_h = params.n_parallel) != null ? _h : 4,
        kv_unified: (_i = params.kv_unified) != null ? _i : true,
        flash_attn: params.flash_attn,
        swa_full: params.swa_full,
        chat_template: params.chat_template,
        jinja: params.jinja,
        reasoning: params.reasoning,
        image_min_tokens: params.image_min_tokens,
        image_max_tokens: params.image_max_tokens,
        warmup: params.warmup,
        no_kv_offload: params.no_kv_offload,
        mmproj_offload: params.mmproj_offload,
        cont_batching: params.cont_batching,
        n_keep: params.n_keep,
        ctx_shift: params.ctx_shift,
        cache_idle_slots: params.cache_idle_slots,
        n_cache_reuse: params.n_cache_reuse,
        lora_paths: params.lora_adapters ? loraPaths : void 0,
        lora_scales: (_j = params.lora_adapters) == null ? void 0 : _j.map((a) => {
          var _a2;
          return (_a2 = a.scale) != null ? _a2 : 1;
        }),
        lora_init_without_apply: params.lora_init_without_apply,
        native_state_paths: params.native_states ? statePaths : void 0,
        spec_draft_model: params.spec_draft_model,
        spec_draft_ngl: params.spec_draft_ngl,
        spec_draft_n_max: params.spec_draft_n_max,
        spec_draft_n_min: params.spec_draft_n_min,
        spec_draft_p_min: params.spec_draft_p_min,
        spec_draft_threads: params.spec_draft_threads,
        spec_draft_threads_batch: params.spec_draft_threads_batch,
        kv_overrides_keys: params.kv_overrides ? Object.keys(params.kv_overrides) : void 0,
        kv_overrides_vals: params.kv_overrides ? Object.values(params.kv_overrides) : void 0,
        // the server's own defaults too, so every request it renders gets them (natlang's /v1/natlang/score items as
        // well as chat), as llama-server's --chat-template-kwargs does; each value travels as its JSON text, which is
        // how the server stores them
        default_template_kwargs_keys: params.default_template_kwargs ? Object.keys(params.default_template_kwargs) : void 0,
        default_template_kwargs_vals: params.default_template_kwargs ? Object.values(params.default_template_kwargs).map((value) => JSON.stringify(value)) : void 0,
        reasoning_budget_tokens: params.reasoning_budget_tokens,
        reasoning_budget_message: params.reasoning_budget_message,
        reasoning_format: params.reasoning_format,
        skip_chat_parsing: params.skip_chat_parsing,
        prefill_assistant: params.prefill_assistant
      });
      const loadedCtxInfo = __spreadProps(__spreadValues({}, loadResult), {
        metadata: {}
      });
      for (let i = 0; i < loadResult.metadata_key.length; i++) {
        loadedCtxInfo.metadata[loadResult.metadata_key[i]] = loadResult.metadata_val[i];
      }
      this.seed = params.seed;
      this.bosToken = loadedCtxInfo.token_bos;
      this.eosToken = loadedCtxInfo.token_eos;
      this.eotToken = loadedCtxInfo.token_eot;
      this.useEmbeddings = !!params.embeddings;
      this.useRerank = params.pooling_type == "rank";
      this.metadata = {
        hparams: {
          nVocab: loadedCtxInfo.n_vocab,
          nCtxTrain: loadedCtxInfo.n_ctx_train,
          nEmbd: loadedCtxInfo.n_embd,
          nLayer: loadedCtxInfo.n_layer
        },
        meta: loadedCtxInfo.metadata
      };
      this.hasEncoder = !!loadedCtxInfo.has_encoder;
      this.decoderStartToken = loadedCtxInfo.token_decoder_start;
      this.addBosToken = loadedCtxInfo.add_bos_token;
      this.addEosToken = loadedCtxInfo.add_eos_token;
      this.chatTemplate = loadedCtxInfo.metadata["tokenizer.chat_template"];
      this.loadedContextInfo = loadedCtxInfo;
      this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
      this.mediaMarker = loadedCtxInfo.media_marker;
      this.chatTemplateKwargs = (_k = params.default_template_kwargs) != null ? _k : {};
      this.logger().debug({ loadedCtxInfo });
    });
  }
  getLoadedContextInfo() {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError("Loaded context info is not available");
    }
    return __spreadValues({}, this.loadedContextInfo);
  }
  //////////////////////////////////////////////
  // High level API
  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  createEmbedding(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      if (!this.useEmbeddings) {
        throw new WllamaError(
          "Embeddings is not enabled. Please set it via LoadModelParams.embeddings"
        );
      }
      const result = yield this.proxy.wllamaAction(
        "embedding",
        {
          _name: "embd_req",
          data_json: JSON.stringify(options),
          files: []
          // TODO: support file input
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(options, false, result.req_id);
    });
  }
  /**
   * Rerank a list of documents against a query.
   * Requires the model to be loaded with embeddings: true and pooling_type: 'rank'.
   * @param options Reranking options (query, documents, top_n)
   * @returns Reranking response with relevance scores sorted highest first
   */
  createRerank(options) {
    return __async(this, null, function* () {
      var _a, _b;
      this.checkModelLoaded();
      if (!this.useEmbeddings || !this.useRerank) {
        throw new WllamaError(
          "Rerank is not enabled. Please set it via LoadModelParams: embeddings = true and pooling_type = rank"
        );
      }
      const top_n = (_a = options.top_n) != null ? _a : options.documents.length;
      let totalTokens = 0;
      const rawResults = [];
      for (let i = 0; i < options.documents.length; i++) {
        const result = yield this.proxy.wllamaAction("rerank", {
          _name: "rrnk_req",
          data_json: JSON.stringify({
            query: options.query,
            document: options.documents[i]
          })
        });
        if (!result.success) {
          throw new WllamaError(
            "Model failed to start reranking",
            "inference_error"
          );
        }
        const { score, tokens_evaluated } = yield this.getSingleResult(result.req_id, "reranking");
        totalTokens += tokens_evaluated;
        rawResults.push({ index: i, score });
      }
      rawResults.sort((a, b) => b.score - a.score);
      return {
        model: (_b = this.getModelMetadata().meta["general.name"]) != null ? _b : "",
        object: "list",
        usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
        results: rawResults.slice(0, top_n).map(({ index, score }) => ({
          index,
          relevance_score: score
        }))
      };
    });
  }
  /**
   * Answer typed questions about a state, using a decision model (TypeSafe-compatible System One API).
   * Image input is not supported yet.
   * @param options The state and the questions
   * @returns The answer of each question
   */
  createSystemOne(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "systemone",
        {
          _name: "sys1_req",
          data_json: JSON.stringify(options)
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start systemone",
          "inference_error"
        );
      }
      return yield this.getSingleResult(
        result.req_id,
        "systemone"
      );
    });
  }
  /**
   * Score candidate replies to chat messages (natlang decision readout): the log-probability of each reply as the closed final assistant message, summed over its tokens after the chat prompt. The replies of an item share the prompt.
   * @param options The items to score
   * @returns Per item, the log-probability and token count of each reply, or the error of an item that could not be rendered
   */
  createNatlangScore(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("score", {
        _name: "scor_req",
        data_json: JSON.stringify(options)
      });
      if (!result.success) {
        throw new WllamaError("Model failed to start scoring", "inference_error");
      }
      return yield this.getSingleResult(
        result.req_id,
        "score"
      );
    });
  }
  /**
   * Render chat messages (and tools) with the chat template, as the server's POST /apply-template does.
   * @param options OAI-compatible chat completion options (messages, tools, add_generation_prompt, ...)
   * @returns The prompt text
   */
  applyTemplate(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "apply_template",
        { _name: "tmpl_req", data_json: JSON.stringify(options) }
      );
      if (!result.success) {
        throw new WllamaError("Failed to apply chat template", "inference_error");
      }
      return result.prompt;
    });
  }
  /**
   * Save the state of a slot (its KV cache, tokens and context checkpoints), as the server's POST /slots/:id?action=save does. The slot must be idle.
   * @param idSlot The slot
   * @returns The saved state, to give to `restoreSlot` of an engine with the same model, build and context settings
   */
  saveSlot(idSlot) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const res = yield this.slotAction(
        idSlot,
        "save",
        filename
      );
      const state = yield this.proxy.slotFileRead(filename);
      return { state, n_saved: res.n_saved, n_written: res.n_written };
    });
  }
  /**
   * Restore a slot from a state of `saveSlot`, as the server's POST /slots/:id?action=restore does. A state that does not fit leaves the slot empty and throws.
   * @param idSlot The slot
   * @param state The saved state
   * @returns The number of restored tokens and bytes read
   */
  restoreSlot(idSlot, state) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        return yield this.slotAction(idSlot, "restore", filename);
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Clear the cached prompt of a slot, as the server's POST /slots/:id?action=erase does.
   * @param idSlot The slot
   */
  eraseSlot(idSlot) {
    return __async(this, null, function* () {
      return yield this.slotAction(idSlot, "erase", "");
    });
  }
  slotAction(idSlot, action, filename) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("slots", {
        _name: "slot_req",
        id_slot: idSlot,
        action,
        filename
      });
      if (!result.success) {
        throw new WllamaError(`Failed to ${action} slot`, "inference_error");
      }
      return yield this.getSingleResult(result.req_id, `slot ${action}`);
    });
  }
  /**
   * Store a natlang native state (natlang's store-file format, natlang.native-state/1) after the model is loaded, as the
   * server's PUT /v1/neuralese/native-states/:id does: the bytes are verified (layout, finite values, content ID), kept as
   * received and, for a state of this model's identity, registered so prompts consume it exactly.
   * @param state The state's bytes
   * @param id The ID the state must have (its content ID); omit to accept any
   * @returns The consumption mode and the state's metadata ({id, ...metadata})
   */
  putNativeState(state, id = "") {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        const result = yield this.proxy.wllamaAction(
          "native_state",
          { _name: "nsta_req", action: "put", id, filename }
        );
        if (!result.success) {
          throw new WllamaError("Failed to put native state", "inference_error");
        }
        return yield this.getSingleResult(result.req_id, "native state put");
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Read a stored native state (one this engine produced, or one put or loaded), as the server's
   * GET /v1/neuralese/native-states/:id does: its bytes exactly as stored, natlang's store-file format.
   * @param id The state's ID
   * @returns The state's bytes and its metadata ({id, ...metadata})
   */
  getNativeState(id) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const result = yield this.proxy.wllamaAction(
        "native_state",
        { _name: "nsta_req", action: "get", id, filename }
      );
      if (!result.success) {
        throw new WllamaError("Failed to get native state", "inference_error");
      }
      const state = yield this.proxy.slotFileRead(filename);
      return { state, meta: JSON.parse(result.meta_json) };
    });
  }
  createChatCompletion(options) {
    return __async(this, null, function* () {
      var _a;
      if (Object.keys(this.chatTemplateKwargs).length > 0) {
        options = __spreadProps(__spreadValues({}, options), {
          chat_template_kwargs: __spreadValues(__spreadValues({}, this.chatTemplateKwargs), (_a = options.chat_template_kwargs) != null ? _a : {})
        });
      }
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  createCompletion(options) {
    return __async(this, null, function* () {
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  /**
   * Private implementation of createCompletion
   */
  createCompletionImpl(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const isStream = !!options.stream;
      const isChat = !!options.messages;
      const customOpt = {};
      if (this.seed !== void 0) {
        customOpt.seed = this.seed;
      }
      let files = [];
      if (isChat) {
        const tmp = this.prepareMultimodalInput(
          options
        );
        options = tmp.params;
        files = tmp.files;
      }
      const result = yield this.proxy.wllamaAction(
        "completion",
        {
          _name: "cmpl_req",
          is_chat: isChat,
          data_json: JSON.stringify(__spreadValues(__spreadValues({}, options), customOpt)),
          files: files.map((f) => new Uint8Array(f))
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(
        options,
        isStream,
        result.req_id
      );
    });
  }
  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   * Only called when stream=true and no onData is provided.
   */
  createCompletionGenerator(options) {
    return new Promise((resolve) => {
      const createGenerator = cbToAsyncIter(
        (callback) => {
          this.createCompletionImpl(__spreadProps(__spreadValues({}, options), {
            onData: (chunk) => callback(chunk)
          })).then(() => callback(void 0, true)).catch((err) => callback(void 0, false, err));
        }
      );
      resolve(createGenerator());
    });
  }
  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality) {
    this.checkModelLoaded();
    if (modality === "image") {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === "audio") {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        "Unsupported modality: " + modality,
        "unknown_error"
      );
    }
  }
  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  exit() {
    return __async(this, null, function* () {
      var _a;
      yield (_a = this.proxy) == null ? void 0 : _a.wllamaExit();
      this.proxy = null;
    });
  }
  /**
   * [FOR DEBUGGING ONLY] Run ggml backend ops tests without loading any model.
   *
   * Initializes the wasm runtime, executes `test-backend-ops` with the given args, then shuts down.
   *
   * For more info, please refer to guides/debug.md
   *
   * @param args Arguments forwarded to test-backend-ops (e.g. ["-o", "ADD"])
   * @returns retcode (0 = all tests passed) and success flag
   */
  testBackendOps() {
    return __async(this, arguments, function* (args = []) {
      var _a;
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (!(yield isSupportMultiThread())) {
        throw new WllamaError(
          "Multi-threading is required to run backend ops tests, but it is not supported in the current environment."
        );
      }
      const tmpProxy = new ProxyToWorker(
        this.getWorkerResources(),
        0,
        // single-thread; no model needed
        (_a = this.config.suppressNativeLog) != null ? _a : false,
        this.logger()
      );
      try {
        yield tmpProxy.moduleInit([]);
        const startResult = yield tmpProxy.wllamaStart();
        if (!startResult.success) {
          throw new WllamaError(
            `Error while calling start function, result = ${startResult}`
          );
        }
        const result = yield tmpProxy.wllamaAction(
          "test_backend_ops",
          { _name: "tbop_req", args: ["test-backend-ops", ...args] }
        );
        return { retcode: result.retcode, success: result.success };
      } finally {
        yield tmpProxy.wllamaExit();
      }
    });
  }
  //////////////////////////////////////////////
  // Low level API
  // TODO: add back
  /**
   * get debug info
   */
  _getDebugInfo() {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      return yield this.proxy.wllamaDebug();
    });
  }
  //////////////////////////////////////////////
  // Utils
  jsonDecode(data_json) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error("Failed to parse JSON:", data_json);
      throw new WllamaError("Failed to parse model output", "inference_error");
    }
  }
  prepareMultimodalInput(params) {
    const msg = params.messages;
    const msgNew = [];
    const files = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent = [];
        for (const c of m.content) {
          if (c.type === "text" || c.type === "native_state") {
            newContent.push(c);
          } else {
            if (!this.mediaMarker) {
              throw new WllamaError(
                "Media marker is undefined",
                "inference_error"
              );
            }
            files.push(c.data);
            newContent.push({
              type: "text",
              text: this.mediaMarker
            });
          }
        }
        msgNew.push(__spreadProps(__spreadValues({}, m), {
          content: newContent
        }));
      } else {
        msgNew.push(m);
      }
    }
    return {
      params: __spreadProps(__spreadValues({}, params), {
        messages: msgNew
      }),
      files
    };
  }
  // release the slot occupied by the request; cancelling an already-finished request is a no-op
  cancelRequest(reqId) {
    return __async(this, null, function* () {
      try {
        yield this.proxy.wllamaAction("cancel", {
          _name: "cncl_req",
          req_id: reqId
        });
      } catch (e) {
        this.logger().warn("Failed to cancel request", reqId, e);
      }
    });
  }
  getSingleResult(reqId, task) {
    return __async(this, null, function* () {
      let completed = false;
      try {
        while (true) {
          const chunk = yield this.proxy.wllamaAction(
            "get_result",
            { _name: "gres_req", req_id: reqId }
          );
          const jsonString = chunk.data_json;
          if (jsonString && jsonString.length > 0) {
            if (chunk.is_error) {
              const jsonData = this.jsonDecode(jsonString);
              throw new WllamaError(
                jsonData.message || `Unknown ${task} error`,
                "inference_error"
              );
            }
            completed = true;
            return this.jsonDecode(jsonString);
          }
          if (!chunk.has_more) {
            completed = true;
            break;
          }
        }
        throw new WllamaError(`No ${task} result received`, "inference_error");
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
    });
  }
  getResponse(options, isStream, reqId) {
    return __async(this, null, function* () {
      var _a, _b;
      let finalResult = null;
      let completed = false;
      try {
        while (true) {
          if ((_a = options.abortSignal) == null ? void 0 : _a.aborted) {
            throw new WllamaAbortError();
          }
          const result_chunk = yield this.proxy.wllamaAction(
            "get_result",
            {
              _name: "gres_req",
              req_id: reqId
            }
          );
          const jsonString = result_chunk.data_json;
          if (!jsonString || jsonString.length === 0) {
            if (!result_chunk.has_more) {
              completed = true;
              break;
            } else {
              continue;
            }
          }
          if (jsonString == "null") {
            continue;
          }
          let jsonData = this.jsonDecode(jsonString);
          finalResult = jsonData;
          if (result_chunk.is_error) {
            this.logger().error("Model returned an error:", jsonData);
            throw new WllamaError(
              jsonData.message || "Unknown inference error",
              "inference_error"
            );
          }
          if (isStream) {
            if (!Array.isArray(jsonData)) {
              jsonData = [jsonData];
            }
            for (const chunk of jsonData) {
              (_b = options.onData) == null ? void 0 : _b.call(options, chunk);
              finalResult = chunk;
            }
          }
          if (!result_chunk.has_more) {
            completed = true;
            break;
          }
        }
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
      return finalResult;
    });
  }
  getWorkerResources() {
    const workerResources = {
      wasmPath: absoluteUrl(this.pathConfig["default"]),
      compat: false
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          "Not using compat mode" + (isFirefox() ? " (expected on Firefox - WebGPU will be disabled)" : "")
        );
      } else {
        const isUsingDefault = this.compat.worker === WasmCompatFromCDN.worker && this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            "Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package."
          );
          this.logger().warn(
            "IMPORTANT: Performance will be significantly degraded in compatibility mode."
          );
        }
        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }
    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }
    return workerResources;
  }
};
export {
  CacheManager,
  LogLevel,
  LoggerWithoutDebug,
  Model,
  ModelManager,
  ModelValidationStatus,
  POLYFILL_ETAG,
  Wllama,
  WllamaAbortError,
  WllamaError,
  WllamaRuntimeError,
  getHFFileSHA256,
  getHFModelSource,
  isValidGgufFile
};
